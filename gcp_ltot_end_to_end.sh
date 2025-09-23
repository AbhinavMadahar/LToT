#!/usr/bin/env bash
set -euo pipefail

# ==============================
# LToT on GCP — one-shot launcher
# ==============================
#
# Usage:
#   bash gcp_ltot_end_to_end.sh /absolute/path/to/src.tar
#
# Optional env overrides:
#   PROJECT_ID, REGION, ZONE, RUN_ID, BUCKET, NUM_70B, NUM_8B, NUM_MIX, HF_TOKEN
#
# Notes:
# - This script assumes you've already authenticated with 'gcloud auth login'.
# - It creates a service account 'ltot-runner' and uses Managed Instance Groups with Spot VMs.
# - vLLM ports (8000/8100/8200) are reachable *only* from the controller over the VPC.
#
# ==============================

# ---- user-tunable defaults ----
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-a}"
RUN_ID="${RUN_ID:-iclr-$(date +%Y%m%d-%H%M%S)}"
BUCKET="${BUCKET:-ltot-iclr-$(date +%Y%m%d)-$RANDOM}"
NUM_70B="${NUM_70B:-6}"         # A100 x4 servers
NUM_8B="${NUM_8B:-4}"           # L4 x1 servers
NUM_MIX="${NUM_MIX:-4}"         # L4 x1 servers
HF_TOKEN="${HF_TOKEN:-}"         # optional; set if you need gated weights

# ---- input: path to src.tar ----
if [[ $# -ge 1 ]]; then
  SRC_TAR="$1"
else
  SRC_TAR="${SRC_TAR:-./src.tar}"
fi

log() { printf "\n\033[1;34m[ltot]\033[0m %s\n" "$*"; }
die() { printf "\n\033[1;31m[ltot]\033[0m %s\n" "$*" >&2; exit 1; }

# ---- sanity checks ----
command -v gcloud >/dev/null || die "gcloud CLI not found. Install Google Cloud SDK first."
command -v gsutil >/dev/null || die "gsutil not found (comes with gcloud)."
[[ -n "${PROJECT_ID}" && "${PROJECT_ID}" != "(unset)" ]] || die "PROJECT_ID not set and no default in gcloud. Run: gcloud config set project <YOUR_PROJECT_ID>"
[[ -f "${SRC_TAR}" ]] || die "src.tar not found at ${SRC_TAR}. Pass absolute path as first arg."

log "Using project=${PROJECT_ID}  region=${REGION}  zone=${ZONE}"
log "Run ID = ${RUN_ID}"
log "Artifacts bucket = gs://${BUCKET}"
log "src.tar = ${SRC_TAR}"

# ---- configure gcloud ----
gcloud config set project "${PROJECT_ID}" >/dev/null
gcloud config set compute/region "${REGION}" >/dev/null
gcloud config set compute/zone "${ZONE}" >/dev/null

# ---- enable required APIs ----
log "Enabling APIs (compute, storage)..."
gcloud services enable compute.googleapis.com storage.googleapis.com >/dev/null

# ---- create bucket & upload code ----
if ! gsutil ls -b "gs://${BUCKET}" >/dev/null 2>&1; then
  log "Creating bucket gs://${BUCKET} in ${REGION}..."
  gsutil mb -p "${PROJECT_ID}" -l "${REGION}" "gs://${BUCKET}"
else
  log "Bucket gs://${BUCKET} already exists; reusing."
fi
log "Uploading code archive..."
gsutil -m cp -n "${SRC_TAR}" "gs://${BUCKET}/runs/${RUN_ID}/code/src.tar"

# ---- service account with least privilege for VMs ----
SA_EMAIL="ltot-runner@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" >/dev/null 2>&1; then
  log "Creating service account ${SA_EMAIL}..."
  gcloud iam service-accounts create ltot-runner --display-name="LToT runner"
fi

# Required for the controller to list/resize MIGs and for VMs to write to GCS logs/artifacts
log "Granting roles to ${SA_EMAIL}..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/compute.viewer" >/dev/null
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/compute.instanceAdmin.v1" >/dev/null
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/storage.objectAdmin" >/dev/null
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/logging.logWriter" >/dev/null

# ---- private firewall: controller -> vLLM only on 8000-8300 ----
log "Ensuring private firewall rule (controller -> vLLM ports 8000-8300)..."
gcloud compute firewall-rules create ltot-vllm-internal \
  --direction=INGRESS --priority=1000 --network=default \
  --action=ALLOW --rules=tcp:8000-8300 \
  --source-tags=ltot-controller --target-tags=ltot-vllm \
  >/dev/null 2>&1 || true

# ---- write startup scripts locally ----
WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

cat > "${WORKDIR}/70b-startup.sh" <<'EOS'
#!/usr/bin/env bash
set -euxo pipefail
# Optional Hugging Face token for gated weights
if [[ -n "${HF_TOKEN:-}" ]]; then
  pip install -U huggingface_hub && huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
fi
pip install -U pip "vllm>=0.6" transformers accelerate
nohup vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 32768 \
  --port 8000 > /var/log/vllm_70b.log 2>&1 &
EOS

cat > "${WORKDIR}/8b-startup.sh" <<'EOS'
#!/usr/bin/env bash
set -euxo pipefail
pip install -U pip "vllm>=0.6" transformers accelerate
nohup vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 32768 \
  --port 8100 > /var/log/vllm_8b.log 2>&1 &
EOS

cat > "${WORKDIR}/mix-startup.sh" <<'EOS'
#!/usr/bin/env bash
set -euxo pipefail
pip install -U pip "vllm>=0.6" transformers accelerate
nohup vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1-AWQ \
  --quantization awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 32768 \
  --port 8200 > /var/log/vllm_mixtral.log 2>&1 &
EOS

cat > "${WORKDIR}/controller-startup.sh" <<'EOS'
#!/usr/bin/env bash
set -euxo pipefail
RUN_ID="${RUN_ID}"
BUCKET="${BUCKET}"
HF_TOKEN="${HF_TOKEN:-}"

# Tools
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip jq

# Workspace
mkdir -p /opt/ltot && cd /opt/ltot
gsutil cp "gs://${BUCKET}/runs/${RUN_ID}/code/src.tar" /opt/ltot/src.tar
tar xf /opt/ltot/src.tar -C /opt/ltot

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip boto3 requests pyyaml "vllm>=0.6"
# If your repo has requirements, install them (ignore failures if not present)
pip install -r /opt/ltot/src/src/requirements.txt || true

# Helper to build endpoints.yaml by querying internal IPs of our MIG instances
eps() {
  local role="$1" port="$2"
  gcloud compute instances list \
    --filter="labels.run_id=${RUN_ID} AND labels.role=${role} AND status=RUNNING" \
    --format="value(INTERNAL_IP)" | awk -v P="$port" '{print "http://"$1":"P}'
}

# Wait a bit for MIG instances to boot
sleep 30

mkdir -p /opt/ltot
cat > /opt/ltot/endpoints.yaml <<YAML
70b:
$(eps 70b 8000 | sed 's/^/  - /')
8b:
$(eps 8b 8100 | sed 's/^/  - /')
mixtral:
$(eps mixtral 8200 | sed 's/^/  - /')
YAML

# Simple readiness check
python - <<'PY'
import time, sys, requests, yaml
with open("/opt/ltot/endpoints.yaml") as f:
    eps = yaml.safe_load(f)
def ready(url, t=1800):
    t0=time.time()
    while time.time()-t0<t:
        try:
            r=requests.get(url.rstrip('/')+"/v1/models", timeout=5)
            if r.status_code==200: return True
        except Exception:
            pass
        time.sleep(5)
    return False
for k,urls in eps.items():
    for u in urls:
        ok=ready(u, 1800 if k=='70b' else 900)
        print(("[ready]" if ok else "[timeout]"), k, u, flush=True)
PY

# Run your harness (equal-median-token budgets; adjust task list as needed)
cd /opt/ltot/src/src
python ../scripts/run_bench.py \
  --controller lt0t \
  --bench gsm_plus,human_eval,math500,mbpp_lite,game24 \
  --budget-median-tokens "8b:700,mixtral:1000,70b:1400" \
  --endpoints-file /opt/ltot/endpoints.yaml \
  --artifacts "gs://${BUCKET}/runs/${RUN_ID}/" \
  --ltot.eta 4 --ltot.micro_probe 1 --ltot.overflow_cap 0.15 \
  --metrics success@1,expansions_to_first_hit,false_promo,width_scaling || true

# Sync artifacts (if your repo writes results/figures here)
gsutil -m rsync -r /opt/ltot/src/src/results "gs://${BUCKET}/runs/${RUN_ID}/results" || true
gsutil -m rsync -r /opt/ltot/src/src/figures "gs://${BUCKET}/runs/${RUN_ID}/figures" || true

# Autoshrink MIGs to zero to stop spend
gcloud compute instance-groups managed resize ltot-8b-mig --size=0 || true
gcloud compute instance-groups managed resize ltot-mix-mig --size=0 || true
gcloud compute instance-groups managed resize ltot-70b-mig --size=0 || true
EOS

# ---- create instance templates ----
log "Creating instance templates..."

# 70B — A100 (A2), tensor-parallel=4
gcloud compute instance-templates create ltot-70b-tpl \
  --machine-type=a2-highgpu-4g \
  --accelerator=count=4,type=nvidia-tesla-a100 \
  --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --service-account="${SA_EMAIL}" \
  --scopes=cloud-platform \
  --labels="run_id=${RUN_ID},role=70b" \
  --tags="ltot-vllm" \
  --metadata="HF_TOKEN=${HF_TOKEN}" \
  --metadata-from-file startup-script="${WORKDIR}/70b-startup.sh" >/dev/null

# 8B — L4 (G2)
gcloud compute instance-templates create ltot-8b-tpl \
  --machine-type=g2-standard-16 \
  --accelerator=count=1,type=nvidia-l4 \
  --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --service-account="${SA_EMAIL}" \
  --scopes=cloud-platform \
  --labels="run_id=${RUN_ID},role=8b" \
  --tags="ltot-vllm" \
  --metadata-from-file startup-script="${WORKDIR}/8b-startup.sh" >/dev/null

# Mixtral — L4 (G2)
gcloud compute instance-templates create ltot-mix-tpl \
  --machine-type=g2-standard-16 \
  --accelerator=count=1,type=nvidia-l4 \
  --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --service-account="${SA_EMAIL}" \
  --scopes=cloud-platform \
  --labels="run_id=${RUN_ID},role=mixtral" \
  --tags="ltot-vllm" \
  --metadata-from-file startup-script="${WORKDIR}/mix-startup.sh" >/dev/null

# ---- create MIGs ----
log "Creating Managed Instance Groups (MIGs)..."
gcloud compute instance-groups managed create ltot-70b-mig \
  --base-instance-name=ltot-70b --size="${NUM_70B}" --template=ltot-70b-tpl >/dev/null

gcloud compute instance-groups managed create ltot-8b-mig \
  --base-instance-name=ltot-8b --size="${NUM_8B}" --template=ltot-8b-tpl >/dev/null

gcloud compute instance-groups managed create ltot-mix-mig \
  --base-instance-name=ltot-mix --size="${NUM_MIX}" --template=ltot-mix-tpl >/dev/null

# ---- create controller VM ----
log "Creating controller VM..."
gcloud compute instances create ltot-controller \
  --machine-type=n2-standard-32 \
  --image-family=debian-12 --image-project=debian-cloud \
  --service-account="${SA_EMAIL}" \
  --scopes=cloud-platform \
  --labels="run_id=${RUN_ID},role=controller" \
  --tags="ltot-controller" \
  --metadata="RUN_ID=${RUN_ID},BUCKET=${BUCKET},HF_TOKEN=${HF_TOKEN}" \
  --metadata-from-file=startup-script="${WORKDIR}/controller-startup.sh" >/dev/null

log "All resources requested. The controller will discover endpoints and start the suite shortly."
log "Artifacts will stream to: gs://${BUCKET}/runs/${RUN_ID}/"
echo
echo "Tips:"
echo "  - Watch MIG instances:   gcloud compute instance-groups managed list-instances ltot-70b-mig"
echo "  - Controller logs:       gcloud compute ssh ltot-controller --journal-tail=100 -q"
echo "  - Stop spend early:      gcloud compute instance-groups managed resize ltot-70b-mig --size=0 && \\ "
echo "                           gcloud compute instance-groups managed resize ltot-8b-mig --size=0 && \\ "
echo "                           gcloud compute instance-groups managed resize ltot-mix-mig --size=0"
