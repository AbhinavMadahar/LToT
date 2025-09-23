# LToT — AWS Plan A Launch Bundle

This bundle launches the **Plan A** topology on AWS (L40S) for your LToT experiments:
- **70B:** 6 replicas on `g6e.12xlarge` (4× L40S each) — **1 on‑demand** (the *fuse*), **5 Spot**.
- **S/M:** 8 single‑GPU `g6e.xlarge` Spot nodes (4× 8B servers on port 8100; 4× Mixtral servers on port 8200).

**Expected outcome** (from our planning):
- **Wall time:** ≈ **17 hours** end‑to‑end (dominated by the 70B slice).
- **Compute cost:** typically **~$510–$570** at ~70% Spot savings (add ~5–10% buffer).

You’ll point your *existing* harness (e.g., `scripts/run_bench.py`) at the endpoints via the provided
endpoint file. All per‑item outputs/checkpoints should stream to S3 so Spot interruptions can be
recovered with minimal loss.

> The controller parameters below (η, micro‑probe, overflow cap, width‑aware bar, short‑circuit)
> follow the LToT defaults in your draft and rely on rung‑level checkpoints / freeze–thaw. fileciteturn0file0

---

## 0) Prereqs (one‑time)
1. **Quotas:** ensure capacity for:
   - On‑Demand: **1× g6e.12xlarge**.
   - Spot: **5× g6e.12xlarge** and **8× g6e.xlarge** (or 2× `g6e.4xlarge` if you do FP16 Mixtral on 2 GPUs).
2. **S3 bucket** for artifacts (e.g., `s3://ltot-iclr/<run-id>/`). Grant read/write to your instance profile.
3. **IAM instance profile** with S3 permissions for the bucket above.
4. **Security group:** allow **SSH (22/tcp)** from your IP; keep vLLM ports private within VPC.
5. **AMI:** latest Deep Learning AMI (Ubuntu 22.04) with CUDA drivers in your region.

---

## 1) Launch
Edit **`launch_aws.sh`** variables at the top (region, AMI, subnet, SG, IAM profile, bucket, run‑id) and run:

```bash
bash launch_aws.sh
```

This will start:
- **1 on‑demand** `g6e.12xlarge` (70B *fuse*).
- **5 Spot** `g6e.12xlarge` (70B replicas).
- **4 Spot** `g6e.xlarge` (8B servers).
- **4 Spot** `g6e.xlarge` (Mixtral servers; AWQ quantized by default).

User‑data cloud‑init scripts (in this bundle) boot vLLM servers automatically:
- **70B:** port **8000** with `--tensor-parallel-size 4`.
- **8B:** port **8100**.
- **Mixtral (AWQ):** port **8200**.

> If you prefer FP16 Mixtral, use the 2‑GPU variant (see the comment in `launch_aws.sh`).

After instances are running, record their **private IPs** and fill **`endpoints.yaml`** accordingly.

---

## 2) Health check
Use:
```bash
python health_check.py endpoints.yaml
```
It queries `/v1/models` on each endpoint and prints readiness.

---

## 3) Run your shards
Your harness decides how to shard; we recommend **256–512 items/shard** and writing results incrementally to S3.
Example invocation (adapt to your paths):
```bash
python scripts/run_bench.py   --controller lt0t   --bench gsm_plus,human_eval,math500,mbpp_lite,game24   --budget-median-tokens "8b:700,mixtral:1000,70b:700,1400,2800"   --endpoints-file endpoints.yaml   --artifacts s3://YOUR_BUCKET/runs/YOUR_RUN/   --ltot.eta 4 --ltot.micro_probe 1 --ltot.overflow_cap 0.15   --metrics success@1,expansions_to_first_hit,false_promo,width_scaling
```

**LToT defaults to match your draft:** `η∈{3,4,5}` (we use **4**), micro‑probe=**1**, overflow cap **ρ∈[0.1,0.2]** (we use **0.15**), width‑aware bar + repeat‑to‑confirm, short‑circuit promotions bound to verifier/exact‑match. fileciteturn0file0

---

## 4) “Never miss the deadline” rule (optional auto‑escalation)
At **T = 8 hours** before your internal must‑finish time, compute remaining 70B tokens **W70**.
Ensure on‑demand replicas satisfy:
```
replicas_on_demand >= ceil( W70 / (6.18e6 tokens/hour * T_hours) )
```
(6.18M tok/h ≈ 1,717.8 tok/s per 4‑GPU L40S replica × 3600). If not, **add on‑demand** replicas until true.

---

## 5) Files in this bundle

- `launch_aws.sh` — AWS CLI launch script (Plan A).
- `user-data-70b.yaml` — cloud‑init for 70B (4× GPU, port 8000).
- `user-data-8b.yaml` — cloud‑init for 8B (port 8100).
- `user-data-mixtral-awq.yaml` — cloud‑init for Mixtral‑AWQ (port 8200).
- `endpoints.yaml` — fill with your private IPs (models → list of `http://ip:port`).
- `health_check.py` — verifies endpoints via OpenAI‑compatible vLLM API.

**Security:** do **not** expose ports 8000/8100/8200 to the public internet; keep them inside VPC.

Good luck — you should be able to kick this off immediately.
