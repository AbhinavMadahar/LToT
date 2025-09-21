#!/usr/bin/env bash
# Vast.ai launcher for LToT: 6× 70B (4×A100 each) + 4× 8B (1×A100) + 4× Mixtral (2×A100)
# Produces endpoints.env to drive your runner with LTOT_BACKEND=openai
set -euo pipefail

### --- 0) EDIT THESE KNOBS ----------------------------------------------------
HF_TOKEN="hf_xxx"             # (optional) Hugging Face token if model is gated
# Bid caps PER INSTANCE HOUR (interruptible). Adjust to taste:
BID_4GPU="2.80"               # ~= $0.70/GPU-h × 4
BID_2GPU="1.40"               # ~= $0.70/GPU-h × 2
BID_1GPU="0.70"               # ~= $0.70/GPU-h × 1

# Models (change if you use different IDs)
MODEL_70B="meta-llama/Meta-Llama-3.1-70B-Instruct"
MODEL_8B="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_MIX="mistralai/Mixtral-8x7B-Instruct-v0.1"

# Paste OFFER IDs you want to rent (from: vastai search offers 'gpu_name=A100 ...')
# Prefer "Secure Cloud" / high reliability hosts near Ashburn/NYC.
OFFERS_4GPU=(1111111 2222222 3333333 4444444 5555555 6666666)          # 6 hosts, each ≥4×A100
OFFERS_1GPU=(7777777 8888888 9999999 1010101)                           # 4 hosts, 1×A100
OFFERS_2GPU=(1212121 1313131 1414141 1515151)                           # 4 hosts, ≥2×A100

# Docker image tag for vLLM's OpenAI-compatible server
VLLM_IMG="vllm/vllm-openai:v0.9.2"
# Disk to attach (GB) — enough to cache weights
DISK_4GPU=220
DISK_2GPU=160
DISK_1GPU=120

### --- 1) Sanity --------------------------------------------------------------
command -v vastai >/dev/null || { echo "Install Vast CLI: pip install -U vastai"; exit 1; }
echo "[i] Using bids: 4GPU=\$${BID_4GPU}/h, 2GPU=\$${BID_2GPU}/h, 1GPU=\$${BID_1GPU}/h"

DATE_TAG=$(date +%y%m%d-%H%M%S)
IDS_FILE="vast_instance_ids.${DATE_TAG}.txt"
EP_FILE="endpoints.env"

touch "$IDS_FILE"
: > "$EP_FILE"

### --- 2) Launch 70B servers (6× 4-GPU) --------------------------------------
i=0
for OFFER in "${OFFERS_4GPU[@]}"; do
  i=$((i+1))
  LABEL="ltot-70b-${i}-${DATE_TAG}"
  echo "[+] Launching 70B replica $i on offer $OFFER (label=$LABEL)"
  vastai create instance "$OFFER" \
    --image "$VLLM_IMG" \
    --disk "$DISK_4GPU" \
    --label "$LABEL" \
    --price "$BID_4GPU" \
    --runtype args \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --args "--model ${MODEL_70B} --tensor-parallel-size 4 --port 8000 --gpu-memory-utilization 0.90 --enable-chunked-prefill --max-num-seqs 1024" \
    | tee -a "$IDS_FILE"
done

### --- 3) Launch 8B servers (4× 1-GPU) ---------------------------------------
i=0
for OFFER in "${OFFERS_1GPU[@]}"; do
  i=$((i+1))
  LABEL="ltot-8b-${i}-${DATE_TAG}"
  echo "[+] Launching 8B server $i on offer $OFFER (label=$LABEL)"
  vastai create instance "$OFFER" \
    --image "$VLLM_IMG" \
    --disk "$DISK_1GPU" \
    --label "$LABEL" \
    --price "$BID_1GPU" \
    --runtype args \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --args "--model ${MODEL_8B} --port 8000 --gpu-memory-utilization 0.90 --enable-chunked-prefill --max-num-seqs 1024" \
    | tee -a "$IDS_FILE"
done

### --- 4) Launch Mixtral servers (4× 2-GPU, TP=2) ----------------------------
i=0
for OFFER in "${OFFERS_2GPU[@]}"; do
  i=$((i+1))
  LABEL="ltot-mix-${i}-${DATE_TAG}"
  echo "[+] Launching Mixtral server $i on offer $OFFER (label=$LABEL)"
  vastai create instance "$OFFER" \
    --image "$VLLM_IMG" \
    --disk "$DISK_2GPU" \
    --label "$LABEL" \
    --price "$BID_2GPU" \
    --runtype args \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --args "--model ${MODEL_MIX} --tensor-parallel-size 2 --port 8000 --gpu-memory-utilization 0.90 --enable-chunked-prefill --max-num-seqs 1024" \
    | tee -a "$IDS_FILE"
done

### --- 5) Collect IPs and build endpoints.env --------------------------------
echo "[i] Waiting 20s for instances to come up..."
sleep 20

# Helper: append all IPs for labels matching a regex to EP_FILE
append_ips () {
  local regex="$1" name="$2"
  # list instances → grep label → pick 'ssh_host' column safely via --raw json if available
  # Fallback: parse table output (column order may vary across CLI versions).
  echo "[i] Discovering ${name} endpoints..."
  # Prefer JSON if supported:
  if vastai list instances --raw json >/dev/null 2>&1; then
    vastai list instances --raw json | python3 - "$regex" "$name" "$EP_FILE" <<'PY'
import json, os, re, sys
rgx, name, outf = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.load(sys.stdin)
ips = []
for inst in data:
    lbl = inst.get("label","")
    if re.search(rgx, lbl):
        ip = inst.get("public_ipaddr") or inst.get("ssh_host") or inst.get("ipaddr")
        port = 8000
        if ip: ips.append(f"http://{ip}:{port}/v1")
with open(outf,"a") as f:
    if ips:
        arr = " ".join(ips)
        f.write(f'export {name}=( {arr} )\n')
        print(f"[OK] {name}: {len(ips)} endpoints")
    else:
        print(f"[WARN] no endpoints for {name}")
PY
  else
    # Table fallback (best-effort): label + ip are usually present
    # Users can tweak if their CLI format differs.
    mapfile -t lines < <(vastai list instances | grep -E "$regex" || true)
    ips=()
    for L in "${lines[@]}"; do
      ip=$(echo "$L" | awk '{for(i=1;i<=NF;i++){if($i ~ /([0-9]{1,3}\.){3}[0-9]{1,3}/){print $i; break}}}')
      [[ -n "${ip:-}" ]] && ips+=("http://${ip}:8000/v1")
    done
    if [[ ${#ips[@]} -gt 0 ]]; then
      echo "export ${name}=( ${ips[*]} )" >> "$EP_FILE"
      echo "[OK] ${name}: ${#ips[@]} endpoints"
    else
      echo "[WARN] no endpoints for $name"
    fi
  fi
}

append_ips "ltot-70b-.*-${DATE_TAG}"   "ENDPOINTS_70B"
append_ips "ltot-8b-.*-${DATE_TAG}"    "ENDPOINTS_8B"
append_ips "ltot-mix-.*-${DATE_TAG}"   "ENDPOINTS_MIX"

cat >> "$EP_FILE" <<'ENV'
# --- Runner toggles ---
export LTOT_BACKEND=openai         # or: local
export OPENAI_API_KEY=dummy        # vLLM usually ignores this
# Example: select a single endpoint (one process → one server)
# export OPENAI_API_BASE=${ENDPOINTS_70B[0]}
ENV

echo
echo "[✓] Wrote endpoints.env with arrays: ENDPOINTS_70B / ENDPOINTS_8B / ENDPOINTS_MIX"
echo "[i] Instance IDs/log were saved to: $IDS_FILE"
echo
echo "Next steps:"
echo "  1) source endpoints.env"
echo "  2) For each endpoint, set OPENAI_API_BASE and run your shard (example below)."
echo
echo "Example dispatch (70B):"
cat <<'EXAMPLE'
# source endpoints.env
for idx in "${!ENDPOINTS_70B[@]}"; do
  export LTOT_BACKEND=openai
  export OPENAI_API_BASE="${ENDPOINTS_70B[$idx]}"
  # Fill your CLI args as usual; one process per endpoint:
  python -m ltot.run run \
    --model llama-3.1-70b-instruct --task gsm_plus --budget 1400 --seed $((idx%3+1)) \
    --out "results/raw/70b.gsm_plus.1400.seed$((idx%3+1)).shard$idx.jsonl" &
done
wait
EXAMPLE

echo
echo "Cleanup (destroy all labeled instances for this launch):"
echo "  vastai list instances | grep ${DATE_TAG} | awk '{print \$1}' | xargs -n1 -I{} vastai destroy instance {}"
