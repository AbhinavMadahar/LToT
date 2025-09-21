# Quickstart to Running the LToT ICLR Experiments

Below is a **self‑contained, step‑by‑step guide** you can hand to your Oxford collaborator. It assumes no prior exposure to ARC specifically, but does assume comfort with Linux, Git, Python, and Slurm. It explains the **why** behind each step (so it’s not just a pile of commands), matches the **paper’s exact experimental protocol**, and shows how to monitor jobs and land a **submission‑ready artifact** even while long‑tail experiments are still running. Where I refer to the experimental design (equal‑compute, width‑scaling $N₀$, ablations, early‑stop, cost law), I’m following your manuscript.  I also reference your collaborator’s background (LLM benchmarking/HPC), so the guide uses practices they’ll already know.&#x20;

---

# LToT on Oxford ARC — A Complete Operator’s Guide

## 1) What you’re running and why it’s organized this way

**Lateral Tree‑of‑Thoughts (LToT)** is a search‑time controller that separates **mainlines** (high‑utility exploitation) from **laterals** (logically consistent, initially low‑utility candidates). It races many laterals cheaply and **promotes** a branch the moment it crosses a **width‑aware mainline bar**; promotions are bound to **verifier‑aligned outcomes** (exact match for math, tests for code). The controller’s core, **LR–SC**, achieves **pseudolinear lateral cost** $Θ(N log N)$ so you can scale lateral coverage without blowing up compute. The paper’s experiments require:

* **Equal‑compute** CoT/ToT/MCTS‑PW/**LToT** at matched median tokens (±2%).
* **Width scaling** over initial lateral width $N₀$.
* **Ablations** (overflow\_off, no\_curvature, no\_width\_bar, no\_short\_circuit, no\_plateau, no\_confirm).
* **Early‑stop** (time‑to‑first‑hit).
* A compact **noisy‑v** study.

All logs and tables roll up into **one artifact** (`results/ltot_artifact.jsonl`) and a **main figure** (`figures/main_equal_compute.svg`).&#x20;

Why this guide emphasizes *ordering and monitoring*: finishing the **ICLR‑core** pieces first (main equal‑compute table, one width‑scaling figure, one ablation table, and early‑stop) is enough for a strong submission; heavier slices (GSM‑Plus, the 70B row, full budget sweeps, full noisy‑v) add depth if they finish in time.&#x20;

---

## 2) ARC in practice (what you need to know in 5 minutes)

* ARC uses **Slurm**. You **never compute on login nodes**; you submit jobs and arrays with `sbatch`.
* Use an **interactive partition** briefly to build environments or validate GPUs (`srun -p interactive --pty /bin/bash`).
* Storage is typically split: a small **\$HOME** for dotfiles, a larger project area (e.g., **\$DATA**) for repos/datasets/results, and an ephemeral **\$SCRATCH/\$TMPDIR** inside jobs—copy results back before exit.
* GPUs are requested with `#SBATCH --gres=gpu:<count>` (optionally constrain to a type like `a100`); choose partitions that make sense for your group’s queue.
* Core Slurm commands you’ll use: `sbatch`, `squeue`, `sinfo`, `sacct`, `scancel`.
  This level of workflow is standard for an LLM benchmarking/HPC operator.&#x20;

---

## 3) Repository orientation (what lives where)

* **Top level**: `README.md`, `requirements.txt`, `Snakefile`, `configs/experiments.yaml`, `launch_arc.sh` (legacy), `ltot/` package.
* **Controllers/algorithms**: `ltot/search/ltot_controller.py`, `ltot/search/lrsc.py`, baselines in `ltot/search/baselines.py`.
* **Data & eval**: `ltot/datasets.py`, `ltot/evaluators.py`.
* **Runner**: `ltot/run.py` (subcommands: `run`, `widthscale`, `ablate`, `earlystop`, `aggregate`).
* **Outputs**: Per‑job JSONLs under `results/raw*`; merged artifact at `results/ltot_artifact.jsonl`; main figure at `figures/main_equal_compute.svg`.
  This layout is already aligned to the paper’s experiment plan.&#x20;

---

## 4) One‑time setup (login or interactive node)

**Why this order?** You prepare a deterministic Python env, put caches on persistent storage to avoid re‑downloading models, and **fail fast** on canonical files before any GPUs are consumed. The paper intentionally **fails closed** if canonical ID lists are missing; catching that now prevents silent cohort drift.&#x20;

1. **Clone and env**

```bash
mkdir -p $DATA/ltot && cd $DATA/ltot
git clone <YOUR_REPO_URL> repo && cd repo
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
```

2. **Model/dataset caches on persistent storage**

```bash
export HF_HOME="${DATA:-$HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
```

3. **Hugging Face access** (accept EULAs for Meta/Mistral once; avoids 401/403 on first load)

```bash
huggingface-cli login
```

4. **Canonical files sanity check** (fast fail if any are missing)

```bash
python - <<'PY'
from ltot.datasets import load_task
for t in ["gsm_hard","math_500","mbpp_lite","game24","humaneval"]:
    try:
        n = sum(1 for _ in load_task(t, seed=1))
        print(f"OK {t}: {n} items")
    except Exception as e:
        print(f"ERROR {t}: {e}")
PY
```

5. **Local smoke test** (validates prompts/verifiers end‑to‑end)

```bash
python -m ltot.run run \
  --model llama-3.1-8b-instruct --task humaneval --budget 350 --seed 1 \
  --out results/smoke.jsonl --shard 0
tail -n +1 results/smoke.jsonl
```

---

## 5) How the job orchestration works (and the small fix you need)

**The idea:** your grid is split into **100 shards** (from `configs/experiments.yaml`) so the cluster can run many independent slices in parallel. The `Snakefile` reads an env var `LTOT_SHARD` to decide which slice a job owns. Each **array task** sets `LTOT_SHARD=${SLURM_ARRAY_TASK_ID}` and then runs Snakemake for *its* share. This yields high throughput while keeping per‑job memory bounded—important for large models. This change is operational only; it does **not** alter experiment semantics from the paper.&#x20;

Create three tiny scripts:

**A) Array worker: `scripts/arc_shard.sbatch`**
Explains: one task = one shard; keeps intermediate JSONLs (`--notemp`) so you can aggregate anytime.

```bash
#!/bin/bash
#SBATCH --job-name=ltot.shard
#SBATCH --partition=short          # adjust to your ARC tenant
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:4               # tune per node type / model mix
#SBATCH --output=logs/arr.%x-%A_%a.out
#SBATCH --error=logs/arr.%x-%A_%a.err
set -euo pipefail
source $PWD/.venv/bin/activate
export LTOT_SHARD=${SLURM_ARRAY_TASK_ID}
echo "[INFO] Running Snakemake for LTOT_SHARD=$LTOT_SHARD"
snakemake --snakefile Snakefile --rerun-incomplete --cores 1 --notemp
```

**B) Optional warmup: `scripts/arc_warmup.sbatch`**
Explains: pre‑loads 8B/Mix models into cache so the arrays don’t all pull weights at once.

```bash
#!/bin/bash
#SBATCH --job-name=ltot.warm
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/warm.%x-%A.out
set -euo pipefail
source $PWD/.venv/bin/activate
python - <<'PY'
from ltot.inference.backends import LocalLM, hf_model_id
for m in ["llama-3.1-8b-instruct","mixtral-8x7b-instruct"]:
    print("WARMUP:", m); LocalLM(hf_model_id(m))
print("Done.")
PY
```

**C) Aggregation: `scripts/arc_aggregate.sbatch`**
Explains: collates any finished shards into the single artifact and main SVG; safe to re‑run.

```bash
#!/bin/bash
#SBATCH --job-name=ltot.aggr
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/aggr.%x-%A.out
set -euo pipefail
source $PWD/.venv/bin/activate
python -m ltot.run aggregate \
  --inputs         results/raw \
  --inputs_width   results/raw_width \
  --inputs_ablate  results/raw_ablate \
  --inputs_latency results/raw_latency \
  --artifact       results/ltot_artifact.jsonl \
  --fig            figures/main_equal_compute.svg
echo "[INFO] Wrote results/ltot_artifact.jsonl and figures/main_equal_compute.svg"
```

---

## 6) Submitting the paper‑exact suite (and how to prioritize what matters)

**Default (everything at once):**

```bash
sbatch scripts/arc_warmup.sbatch
jid=$(sbatch --array=0-99 scripts/arc_shard.sbatch | awk '{print $4}')
sbatch --dependency=afterok:$jid scripts/arc_aggregate.sbatch
```

**Or, stage the “ICLR‑core” first** so the most important tables/figures are guaranteed to land: set these env filters to *restrict* the grid (you can unset them later for the long tail):

* Tasks: `gsm_hard, math_500, humaneval, mbpp_lite, game24`
* Models: `llama-3.1-8b-instruct, mixtral-8x7b-instruct`
* Seeds: `1,2,3`
* Budgets: **Med** tier for each scale (as in the paper)

> Add this tiny snippet once near the top of your `Snakefile` (right after it loads `config`) so the filters work:

```python
import os
task_filter = os.environ.get("LTOT_TASKS","").strip()
if task_filter:
    want = {t.strip() for t in task_filter.split(",") if t.strip()}
    tasks = [t for t in tasks if t in want]
model_filter = os.environ.get("LTOT_MODELS","").strip()
if model_filter:
    wantm = {m.strip() for m in model_filter.split(",") if m.strip()}
    models = [m for m in models if m in wantm]
seed_filter = os.environ.get("LTOT_SEEDS","").strip()
if seed_filter:
    seeds = [int(s) for s in seed_filter.split(",") if s.strip()]
budget_ovr = os.environ.get("LTOT_BUDGETS","").strip()
if budget_ovr:
    ov = {}
    for part in budget_ovr.split(";"):
        if not part: continue
        k,v = part.split("="); ov[k.strip()] = [int(x) if x.isdigit() else x for x in v.split(",")]
    for m in list(budgets.keys()):
        if "all" in ov: budgets[m] = ov["all"]
        if m in ov: budgets[m] = ov[m]
```

Then:

```bash
export LTOT_TASKS="gsm_hard,math_500,humaneval,mbpp_lite,game24"
export LTOT_MODELS="llama-3.1-8b-instruct,mixtral-8x7b-instruct"
export LTOT_SEEDS="1,2,3"
export LTOT_BUDGETS="llama-3.1-8b-instruct=700;mixtral-8x7b-instruct=1000;llama-3.1-70b-instruct=1400"
jid=$(sbatch --array=0-99 scripts/arc_shard.sbatch | awk '{print $4}')
sbatch --dependency=afterok:$jid scripts/arc_aggregate.sbatch
```

**Why this staging helps:** it prioritizes the **main equal‑compute table** and the core **width‑scaling/ablation/early‑stop** results the paper needs, while **GSM‑Plus**, **70B**, and full **budget sweeps**—valuable but not strictly required—can run as time allows.&#x20;

---

## 7) Monitoring and mid‑run validation (so nothing goes “silently wrong”)

**Queues and logs**

```bash
squeue -u $USER
tail -f logs/arr.ltot.shard-*.out
```

**Aggregate anytime** (safe to re‑run repeatedly while shards finish)

```bash
python -m ltot.run aggregate \
  --inputs results/raw --inputs_width results/raw_width \
  --inputs_ablate results/raw_ablate --inputs_latency results/raw_latency \
  --artifact results/ltot_artifact.jsonl --fig figures/main_equal_compute.svg
```

**Equal‑compute parity sentinel (±2% as in the paper)**
If this exits non‑zero, at least one method drifted; re‑run that slice before writing the table.&#x20;

```bash
jq -r 'select(.kind=="fairness") | [.task,.model,.method,(.error_pp|tonumber)]|@tsv' \
  results/ltot_artifact.jsonl | awk '($4<-2 || $4>2){bad=1} END{exit bad}'
```

**Coverage sentinel** (are the core slices present yet?)

```bash
python - <<'PY'
import glob, re
need=[("gsm_hard","llama-3.1-8b-instruct"),("math_500","llama-3.1-8b-instruct"),
      ("humaneval","llama-3.1-8b-instruct"),("mbpp_lite","llama-3.1-8b-instruct"),
      ("gsm_hard","mixtral-8x7b-instruct"),("math_500","mixtral-8x7b-instruct")]
have=set()
for p in glob.glob("results/raw/*.jsonl"):
    m=re.search(r"results/raw/([^\.]+)\.([^\.]+)\.", p)
    if m: have.add((m.group(2), m.group(1)))
missing=[x for x in need if x not in have]
print("MISSING:", missing); exit(1 if missing else 0)
PY
```

**Why these checks:** the paper’s fairness rule and dataset discipline mean most manuscript‑impacting issues are **visible early**; these sentinels keep the chance of a “silent” unusable slice very low.&#x20;

---

## 8) Reading the outputs (so you know when you’re submission‑ready)

* **Per‑job JSONLs** land in:

  * `results/raw/…` (main runs), `results/raw_width/…`, `results/raw_ablate/…`, `results/raw_latency/…`
* **The artifact** `results/ltot_artifact.jsonl` contains:

  * `run` records for each (task, model, method, budget, seed).
  * `metric` records (Success\@1/Pass\@1 with CI, fairness parity, cost‑law fit, etc.).
  * `promotion_event` diagnostics, `rung_costs`, and early‑stop latency (if enabled).
* **The figure** `figures/main_equal_compute.svg` summarizes the main equal‑compute table.

These match the tables/plots described in the manuscript (equal‑compute table, width‑scaling curves, ablations, early‑stop).&#x20;

---

## 9) Troubleshooting (fast triage that covers 90% of issues)

* **Hugging Face 401/403** at model load → log in; ensure EULAs accepted.
* **Canonical file missing** → place it at the path from `configs/experiments.yaml` (the repo is fail‑closed by design).&#x20;
* **OOM on 70B** → schedule those jobs on fast, high‑memory GPUs and keep `--gres=gpu:4` (the 70B row is optional for the first submission).&#x20;
* **Disk/quota** → set caches to `$DATA` (see §4), keep `--notemp`, aggregate often, clean as needed.
* **Fairness guard trips** → re‑run that method/task at the calibrated knob; the ±2% rule is part of the paper’s parity spec.&#x20;

---

## 10) Committing results to Git (what to include and why)

**Commit these (small, reviewable):**

* `results/ltot_artifact.jsonl` (single source for all tables/metrics/diagnostics).
* `figures/main_equal_compute.svg` (main figure).
* `results/run_metadata.json` (provenance: commit hash, Python/pkg versions).

**Do not commit** shard raws/logs. Add to `.gitignore`:

```gitignore
results/raw*
results/raw_width*
results/raw_ablate*
results/raw_latency*
logs/*
```

**Provenance stamp**

```bash
python - <<'PY'
import json, subprocess, platform, time
sh=lambda x: subprocess.check_output(x, shell=True, text=True).strip()
meta={"timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
      "git_commit":sh("git rev-parse --short HEAD"),
      "python":platform.python_version(),
      "pip_freeze":sh("python -m pip freeze"),
      "torch":__import__("torch").__version__,
      "transformers":__import__("transformers").__version__}
json.dump(meta, open("results/run_metadata.json","w"), indent=2)
print("Wrote results/run_metadata.json")
PY

git add figures/main_equal_compute.svg results/ltot_artifact.jsonl results/run_metadata.json .gitignore
git commit -m "ICLR artifact & main figure (+ provenance)"
git push
```

**Why this subset:** it’s everything a reviewer needs to reproduce the reported tables/figure without flooding the repo with large, noisy intermediates. The artifact can always be re‑created from raws if you keep those elsewhere.

---

## 11) When time is tight (the “ICLR‑core” cut)

If wall‑time is at risk, ensure the artifact contains:

* **Main equal‑compute table** across math, code, and ToT‑style puzzles.
* **One width‑scaling** sweep (e.g., on MATH‑500).
* **One ablation table** (e.g., on MATH‑500).
* **Early‑stop** latency (e.g., MATH‑500 and/or HumanEval).

These four deliver the paper’s main claims; GSM‑Plus, full budget sweeps, 70B, and full noisy‑v can land later without undermining the submission.&#x20;

---

### Final notes tailored to your collaborator

* The controller’s *operational logic* (LR–SC, width‑aware bar with confirm, verifier‑bound promotion, plateau trigger) is **already encoded** in the repo; the orchestration above simply gives ARC the right shape of work.&#x20;
* Given his LLM benchmarking/CUDA/HPC background, watching `squeue`, tailing array logs, and re‑aggregating periodically will feel familiar—and is enough to keep the risk of “silent” manuscript issues minimal.&#x20;

If you’d like, I can package the three `scripts/*.sbatch` files plus the `Snakefile` env‑filter snippet as a single patch you can `git apply`—but the guide above is fully runnable as‑is.
