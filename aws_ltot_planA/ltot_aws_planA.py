#!/usr/bin/env python3
""" 
LToT — AWS Plan A single-shot launcher (L40S, vLLM)

What this script does (single command):
  1) Launches AWS EC2 instances for Plan A:
     - 70B: 6 replicas on g6e.12xlarge (4× L40S each): 1 on-demand (the "fuse"), 5 Spot
     - S/M: 8 × g6e.xlarge Spot (4× 8B servers, 4× Mixtral servers; Mixtral defaults to AWQ)
  2) Waits for instances to pass status checks
  3) Writes endpoints.yaml with the private IP:port list for each model pool
  4) Performs a /v1/models readiness check on every endpoint
  5) Prints a ready-to-copy harness command template (you can paste your own)

Prereqs on your machine:
  pip install boto3 requests pyyaml

AWS prerequisites on your account (do once):
  - Quotas: On-Demand 1× g6e.12xlarge; Spot 5× g6e.12xlarge + 8× g6e.xlarge
  - An S3 bucket/prefix for artifacts, e.g. s3://ltot-iclr/runs/<run-id>/
  - IAM instance profile that can read/write that S3 prefix
  - Security group allowing SSH from you; keep vLLM ports private inside VPC

Security: the vLLM servers are bound to the instance and intended for intra-VPC access only.
Do NOT expose ports 8000/8100/8200 publicly.

Notes: This script follows the controller defaults and equal-median-token budgeting used in your draft
(LR–SC with short-circuit; width-aware bar; repeat-to-confirm). It simply provisions and wires
the serving layer; your harness drives the actual experiments.  fileciteturn0file0
"""

import argparse, sys, time, os, textwrap, datetime
import boto3
from botocore.exceptions import ClientError, WaiterError
import yaml
import requests

# ---------- User-data (cloud-init) payloads ----------

def make_user_data_70b(s3_prefix: str, hf_token: str|None):
    return f"""#cloud-config
runcmd:
  - bash -lc 'set -euxo pipefail
    export LTOT_S3_PREFIX="{s3_prefix}"
    {f'export HF_TOKEN="{hf_token}"' if hf_token else ''}
    sudo apt-get update -y
    pip install -U pip "vllm>=0.6" transformers accelerate datasets boto3
    if [ -n "${{HF_TOKEN:-}}" ]; then huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential; fi
    nohup vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct       --tensor-parallel-size 4       --max-model-len 8192       --gpu-memory-utilization 0.90       --enable-chunked-prefill --enable-prefix-caching       --max-num-batched-tokens 32768       --port 8000 > /var/log/vllm_70b.log 2>&1 &
    cat >/usr/local/bin/spot-guard.sh <<EOS
#!/usr/bin/env bash
set -euo pipefail
META=http://169.254.169.254/latest/meta-data/spot/instance-action
while true; do
  if timeout 1 curl -fsS "$META" >/tmp/instance-action.json 2>/dev/null; then
    echo "[guard] spot reclaim notice: $(cat /tmp/instance-action.json)" | tee -a /var/log/spot-guard.log
    if [ -n "${LTOT_S3_PREFIX:-}" ]; then aws s3 sync /opt/ltot/out "$LTOT_S3_PREFIX/$(hostname)/" --exclude "*" --include "*.json" --include "*.ndjson" --include "*.csv" || true; fi
    pkill -TERM -f "vllm serve" || true
    sleep 25
    exit 0
  fi
  sleep 5
done
EOS
    chmod +x /usr/local/bin/spot-guard.sh
    nohup /usr/local/bin/spot-guard.sh >/var/log/spot-guard.log 2>&1 &
    mkdir -p /opt/ltot/out
  '
"""


def make_user_data_8b(s3_prefix: str):
    return f"""#cloud-config
runcmd:
  - bash -lc 'set -euxo pipefail
    export LTOT_S3_PREFIX="{s3_prefix}"
    sudo apt-get update -y
    pip install -U pip "vllm>=0.6" transformers accelerate datasets boto3
    nohup vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct       --max-model-len 8192       --gpu-memory-utilization 0.90       --enable-chunked-prefill --enable-prefix-caching       --max-num-batched-tokens 32768       --port 8100 > /var/log/vllm_8b.log 2>&1 &
    mkdir -p /opt/ltot/out
  '
"""


def make_user_data_mixtral_awq(s3_prefix: str, model_id: str):
    return f"""#cloud-config
runcmd:
  - bash -lc 'set -euxo pipefail
    export LTOT_S3_PREFIX="{s3_prefix}"
    sudo apt-get update -y
    pip install -U pip "vllm>=0.6" transformers accelerate datasets boto3
    nohup vllm serve {model_id}       --quantization awq       --max-model-len 8192       --gpu-memory-utilization 0.95       --enable-chunked-prefill --enable-prefix-caching       --max-num-batched-tokens 32768       --port 8200 > /var/log/vllm_mixtral.log 2>&1 &
    mkdir -p /opt/ltot/out
  '
"""


def make_user_data_mixtral_fp16_2gpu(s3_prefix: str, base_id: str):
    return f"""#cloud-config
runcmd:
  - bash -lc 'set -euxo pipefail
    export LTOT_S3_PREFIX="{s3_prefix}"
    sudo apt-get update -y
    pip install -U pip "vllm>=0.6" transformers accelerate datasets boto3
    nohup vllm serve {base_id}       --tensor-parallel-size 2       --max-model-len 8192       --gpu-memory-utilization 0.92       --enable-chunked-prefill --enable-prefix-caching       --max-num-batched-tokens 32768       --port 8200 > /var/log/vllm_mixtral.log 2>&1 &
    mkdir -p /opt/ltot/out
  '
"""


# ---------- EC2 helpers ----------

def launch_instances(ec2, *, image_id, instance_type, count, spot, user_data, subnet_id, sg_ids, key_name, iam_profile_name, name_tag, run_id, placement=None):
    tags = [
        {'Key': 'Name', 'Value': name_tag},
        {'Key': 'RunId', 'Value': run_id},
    ]
    params = dict(
        ImageId=image_id,
        InstanceType=instance_type,
        MinCount=count,
        MaxCount=count,
        KeyName=key_name,
        SubnetId=subnet_id,
        SecurityGroupIds=sg_ids,
        IamInstanceProfile={'Name': iam_profile_name},
        TagSpecifications=[{'ResourceType': 'instance', 'Tags': tags}],
        UserData=user_data,
    )
    if spot:
        params['InstanceMarketOptions'] = {'MarketType': 'spot', 'SpotOptions': {'InstanceInterruptionBehavior': 'terminate'}}
    if placement:
        params['Placement'] = placement

    r = ec2.run_instances(**params)
    ids = [i['InstanceId'] for i in r['Instances']]
    return ids


def wait_instances_ok(ec2, ids, region):
    if not ids:
        return
    print(f"[+] Waiting for {len(ids)} instance(s) to enter 'running'...")
    waiter = ec2.get_waiter('instance_running')
    waiter.wait(InstanceIds=ids)
    print(f"[+] Waiting for instance status checks to pass (ok)...")
    waiter2 = ec2.get_waiter('instance_status_ok')
    waiter2.wait(InstanceIds=ids)
    print(f"[+] Instances are running and passed health checks.")


def get_private_ips(ec2, ids):
    if not ids:
        return []
    desc = ec2.describe_instances(InstanceIds=ids)
    ips = []
    for res in desc['Reservations']:
        for inst in res['Instances']:
            ip = inst.get('PrivateIpAddress')
            if ip:
                ips.append(ip)
    return ips


def write_endpoints_yaml(path, seventyb_ips, eightb_ips, mix_ips):
    data = {
        'models': {
            '70b': [f"http://{ip}:8000" for ip in seventyb_ips],
            '8b':  [f"http://{ip}:8100" for ip in eightb_ips],
            'mixtral': [f"http://{ip}:8200" for ip in mix_ips],
        }
    }
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"[+] Wrote endpoints to {path}")
    return data


def readiness_check(endpoints, timeout_per=1800):
    """Poll /v1/models for each endpoint until 200 or timeout_per seconds (per group)."""
    def ok(u):
        try:
            r = requests.get(u.rstrip('/') + '/v1/models', timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    all_ok = True
    for group, urls in endpoints['models'].items():
        start = time.time()
        pending = set(urls)
        print(f"[+] Checking readiness for group '{group}' ({len(urls)} endpoints)...")
        last_report = 0
        while pending and (time.time() - start) < timeout_per:
            for u in list(pending):
                if ok(u):
                    print(f"    ready: {u}")
                    pending.remove(u)
            now = time.time()
            if now - last_report > 15:
                print(f"    waiting on {len(pending)} endpoint(s)...")
                last_report = now
            time.sleep(5)
        if pending:
            print(f"[!] Timed out waiting for: {sorted(pending)}")
            all_ok = False
        else:
            print(f"[+] Group '{group}' ready.")
    return all_ok


# ---------- Main orchestration ----------

def main():
    ap = argparse.ArgumentParser(description="LToT — AWS Plan A launcher (single script)")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--ami-id", required=True, help="Deep Learning AMI (Ubuntu 22.04) with CUDA, e.g., ami-...")
    ap.add_argument("--subnet-id", required=True)
    ap.add_argument("--sg-id", required=True, nargs='+', help="One or more security group IDs (space-separated)")
    ap.add_argument("--iam-instance-profile", required=True, help="Instance profile NAME with S3 access")
    ap.add_argument("--key-name", required=True, help="EC2 key pair name")
    ap.add_argument("--s3-prefix", required=True, help="s3://bucket/prefix for artifacts/checkpoints")
    ap.add_argument("--run-id", default=f"run-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Optional HF token for gated models")
    ap.add_argument("--mixtral-mode", choices=["awq","fp16-2gpu"], default="awq")
    ap.add_argument("--mixtral-awq-model", default="mistralai/Mixtral-8x7B-Instruct-v0.1-AWQ")
    ap.add_argument("--mixtral-base-model", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    ap.add_argument("--seventyB-replicas", type=int, default=6, help="Total 70B replicas (4 GPUs each)")
    ap.add_argument("--seventyB-spot", type=int, default=5, help="How many of the 70B replicas are Spot (rest on-demand)")
    ap.add_argument("--eightB-servers", type=int, default=4, help="Number of 8B single-GPU servers (g6e.xlarge)")
    ap.add_argument("--mixtral-servers", type=int, default=4, help="Number of Mixtral servers (AWQ: g6e.xlarge; FP16: g6e.4xlarge/2GPU)")
    ap.add_argument("--endpoints-out", default="endpoints.yaml")
    ap.add_argument("--skip-wait", action="store_true", help="Skip readiness polling")
    args = ap.parse_args()

    if args.seventyB_spot > args.seventyB_replicas:
        print("--seventyB-spot cannot exceed --seventyB-replicas", file=sys.stderr)
        sys.exit(2)
    fuse_on_demand = args.seventyB_replicas - args.seventyB_spot
    if fuse_on_demand < 1:
        print("[i] For safety we enforce at least 1 on-demand 70B replica; increasing on-demand by 1.")
        args.seventyB_spot -= 1
        fuse_on_demand = 1

    print(f"[Plan] Region={args.region}  RunId={args.run_id}")
    print(f"       70B: replicas={args.seventyB_replicas} (Spot={args.seventyB_spot}, OnDemand={fuse_on_demand}) on g6e.12xlarge")
    print(f"       8B:  {args.eightB_servers} × g6e.xlarge")
    if args.mixtral_mode == "awq":
        print(f"       Mixtral: {args.mixtral_servers} × g6e.xlarge (AWQ model: {args.mixtral_awq_model})")
    else:
        print(f"       Mixtral: {args.mixtral_servers} × g6e.4xlarge (FP16, 2 GPUs) base model: {args.mixtral_base_model}")

    session = boto3.session.Session(region_name=args.region)
    ec2 = session.client("ec2")

    # 1) Launch 70B on-demand "fuse"
    user_data_70b = make_user_data_70b(args.s3_prefix, args.hf_token)
    ondemand_ids = []
    if fuse_on_demand > 0:
        print(f"[+] Launching {fuse_on_demand} on-demand 70B replica(s)...")
        ondemand_ids = launch_instances(ec2,
            image_id=args.ami_id, instance_type="g6e.12xlarge", count=fuse_on_demand, spot=False,
            user_data=user_data_70b, subnet_id=args.subnet_id, sg_ids=args.sg_id, key_name=args.key_name,
            iam_profile_name=args.iam_instance_profile, name_tag="ltot-70b-fuse-ondemand", run_id=args.run_id)

    # 2) Launch 70B Spot replicas
    spot_ids_70b = []
    if args.seventyB_spot > 0:
        print(f"[+] Launching {args.seventyB_spot} Spot 70B replica(s)...")
        spot_ids_70b = launch_instances(ec2,
            image_id=args.ami_id, instance_type="g6e.12xlarge", count=args.seventyB_spot, spot=True,
            user_data=user_data_70b, subnet_id=args.subnet_id, sg_ids=args.sg_id, key_name=args.key_name,
            iam_profile_name=args.iam_instance_profile, name_tag="ltot-70b-spot", run_id=args.run_id)

    # 3) Launch 8B servers (Spot)
    print(f"[+] Launching {args.eightB_servers} Spot 8B single-GPU server(s)...")
    user_data_8b = make_user_data_8b(args.s3_prefix)
    spot_ids_8b = []
    if args.eightB_servers > 0:
        spot_ids_8b = launch_instances(ec2,
            image_id=args.ami_id, instance_type="g6e.xlarge", count=args.eightB_servers, spot=True,
            user_data=user_data_8b, subnet_id=args.subnet_id, sg_ids=args.sg_id, key_name=args.key_name,
            iam_profile_name=args.iam_instance_profile, name_tag="ltot-8b-spot", run_id=args.run_id)

    # 4) Launch Mixtral (Spot)
    mix_ids = []
    if args.mixtral_servers > 0:
        if args.mixtral_mode == "awq":
            print(f"[+] Launching {args.mixtral_servers} Spot Mixtral-AWQ single-GPU server(s)...")
            user_data_mix = make_user_data_mixtral_awq(args.s3_prefix, args.mixtral_awq_model)
            mix_ids = launch_instances(ec2,
                image_id=args.ami_id, instance_type="g6e.xlarge", count=args.mixtral_servers, spot=True,
                user_data=user_data_mix, subnet_id=args.subnet_id, sg_ids=args.sg_id, key_name=args.key_name,
                iam_profile_name=args.iam_instance_profile, name_tag="ltot-mixtral-awq-spot", run_id=args.run_id)
        else:
            nodes = args.mixtral_servers
            print(f"[+] Launching {nodes} Spot Mixtral FP16 (2 GPUs) server node(s)...")
            user_data_mix = make_user_data_mixtral_fp16_2gpu(args.s3_prefix, args.mixtral_base_model)
            mix_ids = launch_instances(ec2,
                image_id=args.ami_id, instance_type="g6e.4xlarge", count=nodes, spot=True,
                user_data=user_data_mix, subnet_id=args.subnet_id, sg_ids=args.sg_id, key_name=args.key_name,
                iam_profile_name=args.iam_instance_profile, name_tag="ltot-mixtral-fp16-spot", run_id=args.run_id)

    all_ids = ondemand_ids + spot_ids_70b + spot_ids_8b + mix_ids
    print(f"[i] Requested instances: {len(all_ids)} total")
    if not all_ids:
        print("[!] No instances were launched. Check quotas and parameters.")
        sys.exit(1)

    # Wait for all instances to be running & healthy
    try:
        wait_instances_ok(ec2, all_ids, args.region)
    except WaiterError as e:
        print(f"[!] Waiter error: {e}. Proceeding to fetch whatever is up.")

    # Collect IPs
    ips_ondemand = get_private_ips(ec2, ondemand_ids)
    ips_70b_spot = get_private_ips(ec2, spot_ids_70b)
    ips_8b = get_private_ips(ec2, spot_ids_8b)
    ips_mix = get_private_ips(ec2, mix_ids)
    ips_70b_all = ips_ondemand + ips_70b_spot

    print(f"[+] 70B endpoints: {ips_70b_all}")
    print(f"[+] 8B  endpoints: {ips_8b}")
    print(f"[+] Mix endpoints: {ips_mix}")

    endpoints = write_endpoints_yaml(args.endpoints_out, ips_70b_all, ips_8b, ips_mix)

    if not args.skip_wait:
        print("[i] Polling endpoints for readiness (70B model load can take several minutes after boot)...")
        ok = readiness_check(endpoints, timeout_per=1800)
        if not ok:
            print("[!] Some endpoints did not report ready within the polling window. You can re-run readiness later with:")
            print("""    python - <<'PY'
import yaml,requests,time
cfg=yaml.safe_load(open('endpoints.yaml'))
for g,urls in cfg['models'].items():
  print('\n['+g+']')
  for u in urls:
    try:
      r=requests.get(u.rstrip('/')+'/v1/models',timeout=5)
      print(' ',u, r.status_code)
    except Exception as e:
      print(' ',u,'ERR',e)
PY""")

    # Print harness template
    print("\n=== Ready. Example harness invocation (edit to your paths) ===\n")
    print(textwrap.dedent(f"""
      python scripts/run_bench.py \
        --controller lt0t \
        --bench gsm_plus,human_eval,math500,mbpp_lite,game24 \
        --budget-median-tokens "8b:700,mixtral:1000,70b:700,1400,2800" \
        --endpoints-file {args.endpoints_out} \
        --artifacts {args.s3_prefix.rstrip('/')}/{args.run_id}/ \
        --ltot.eta 4 --ltot.micro_probe 1 --ltot.overflow_cap 0.15 \
        --metrics success@1,expansions_to_first_hit,false_promo,width_scaling
    """).strip())

if __name__ == "__main__":
    try:
        main()
    except ClientError as e:
        print(f"[AWS ERROR] {e}", file=sys.stderr)
        sys.exit(1)
