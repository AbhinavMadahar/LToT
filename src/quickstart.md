# Quickstart to Running the LToT Experiments on Oxford ARC

This is a quickstart for running the Lateral Tree-of-Thoughts (LToT) experiments on the Oxford ARC HPC cluster.
It covers: (1) getting onto Oxford ARC, (2) one‑time setup for local‑LM, Hugging Face–based runs, (3) launching the full experiment suite with **one command** via **Snakemake**, (4) monitoring and recovering jobs, and (5) committing results (with Git LFS). Defaults and controller behavior align with the LToT manuscript you shared (notably §4.2–§4.6 on LR‑SC, promotion, bars, and plateau logic).&#x20;

---

# Running the LToT Experiments on **Oxford ARC**

> **Scope**
>
> * Start all experiments with a single command.
> * Check live status on ARC (Slurm) and see what has finished.
> * Ensure all outputs (metrics, traces, and figures) land in **one artifact file** + reproducible figure bundle.
> * Commit results back to the Git repository (with Git LFS).
> * Uses **local LMs via Hugging Face**; no external API calls required.

**Context & provenance**

* The controller, parameters, and run protocol follow the Lateral Tree‑of‑Thoughts (LToT) manuscript included in this repo; defaults (η, ρ, micro‑probes, width‑aware bars, short‑circuiting, plateau trigger) are set to those in **§4.2–§4.6**.&#x20;
* Dr. **Emanuele La Malfa** (Oxford, Dept. of CS) is a postdoc collaborator on the LToT project; he has deep experience with LM benchmarking and explainability, which should make ARC onboarding and this pipeline straightforward.&#x20;
* Some experiments reference fixed‑point/stability ideas from the *Fixed Point Explainability* preprint; the repo’s analysis scripts include an optional “fixed‑point check” for ablations.&#x20;

---

## 0) What you’ll run / what you’ll get

* **What runs:** the full LToT experimental suite (math/code/ToT‑puzzle benchmarks) with our **LR‑SC** lateral controller, narrow mainlines, width‑aware thresholds, and verifier‑bound promotion; budgets and seeds are set from `config/*.yaml`.&#x20;
* **Single command:** `snakemake --profile profiles/arc all`
* **Outputs:**

  * **`results/artifact.parquet`** — one consolidated artifact (all items, seeds, metrics, promotions, rung traces, budgets, RNG states).
  * **`results/figures/`** — deterministic figure bundle; `results/figures.pdf` is a stitched PDF.
  * **`results/summary.json`** — compact top‑line metrics for CI and paper sync.
* **Paper sync:** `snakemake paper` (or `snakemake all`) materializes LaTeX inputs and figures so ChatGPT can read `results/artifact.parquet` and update the `.tex` accordingly.

---

## 1) Oxford ARC: 5‑minute orientation

ARC is an HPC service with **Slurm**. You’ll connect to a **login node**, prepare your environment, and submit **jobs** that run on **compute nodes**. (Names of partitions/QoS vary—use the discovery commands below to see what *your* account can use.)

### Access & filesystem

* **SSH:** `ssh <oxford-username>@arc-login.arc.ox.ac.uk` (example hostname; use your actual ARC login host).
* **Storage tips:**

  * `$HOME` — small, backed‑up. Good for code, envs.
  * `$SCRATCH` or project space — large, fast. Put **models**, datasets, and run outputs here.
* **Modules/conda:** ARC typically provides environment modules. We use **conda/mamba** in this guide.

### Slurm primers

* **Discover capacity:**

  ```bash
  sinfo -o "%P %G %m %c %f %N"       # partitions, gpus, mem, cores
  sacctmgr show qos format=Name%15,Priority,MaxTRES,MaxWall,MaxJobs
  ```
* **Your jobs:**

  ```bash
  squeue -u $USER
  sacct -u $USER --format=JobID%18,State,Elapsed,MaxRSS,ReqTRES%40
  scancel <jobid>
  ```

> **Note**
> Partition names, QoS tiers, and GPU labels differ by account/allocation and change over time. Use the `sinfo/sacctmgr` commands above to select the best GPU partition you can access (A100/H100/L40/A40/V100 etc.). Keep `$SCRATCH` for heavy I/O.

---

## 2) One‑time setup (15–25 minutes of human time)

> Do this once on a login node. After that, running is one command.

### 2.1 Clone and layout

```bash
# On ARC login node
mkdir -p $HOME/src && cd $HOME/src
git clone <YOUR_REPO_URL> ltots
cd ltots
```

Recommended dirs (created by the repo on first run):

```
config/            # *.yaml (budgets, model refs, benchmark lists)
profiles/arc/      # Snakemake Slurm profile (cluster config)
envs/              # conda env files
models/            # symlinks → $SCRATCH/models (large)
data/              # symlinks → $SCRATCH/data
results/           # symlinks → $SCRATCH/results
logs/              # symlinks → $SCRATCH/logs
```

Create large‑storage targets and link them:

```bash
mkdir -p $SCRATCH/{models,data,results,logs}
ln -sfn $SCRATCH/models  models
ln -sfn $SCRATCH/data    data
ln -sfn $SCRATCH/results results
ln -sfn $SCRATCH/logs    logs
```

### 2.2 Conda env

```bash
# If conda not installed, load module or install micromamba in $HOME
module avail 2>/dev/null | grep -i conda || true
# Create env
conda env create -f envs/ltot.yaml
conda activate ltot
python -V
```

### 2.3 Hugging Face (offline‑friendly)

We run **local LMs via HF Transformers**. ARC often blocks outbound network on compute nodes, so **prefetch** models on a login node (or on your laptop, then `rsync` to ARC).

1. **Authenticate (if models are gated):**

```bash
python -m pip install "huggingface_hub[cli]"
huggingface-cli login   # pastes an access token
```

2. **Prefetch models/datasets** (examples; adjust to your plan):

```bash
# 8B instruct
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
  --local-dir $SCRATCH/models/llama3-8b-instruct --local-dir-use-symlinks False

# Mixtral 8x7B instruct (Mixture-of-Experts; needs gating on some mirrors)
huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --local-dir $SCRATCH/models/mixtral-8x7b-instruct --local-dir-use-symlinks False

# Optionally smaller distilled baselines, tokenizers, or math verifier heads
```

3. **Offline flags** (set in your shell and the Snakemake profile):

```bash
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

> If your allocation requires, place secrets in `config/secrets.env` (read by Snakemake without committing plaintext):
>
> ```
> HF_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxx
> WANDB_API_KEY=  # optional; we default to local logging
> ```
>
> The pipeline reads any needed credentials from the environment only; nothing is hard‑coded. (**You asked for this behavior; it’s implemented.**)

### 2.4 Snakemake profile for ARC (pre‑wired)

`profiles/arc/cluster.yaml` maps Snakemake resources → Slurm (`sbatch`) and lets each rule request GPUs/CPU/mem/time. You don’t need to write job scripts; Snakemake will do it.

To see key defaults:

```bash
yq . profiles/arc/cluster.yaml
yq . profiles/arc/config.yaml
```

---

## 3) **Run everything (single command)**

From the repo root (env activated):

```bash
snakemake --profile profiles/arc all
```

What it does:

* Uses the LToT controller with the **dual‑score frontier**, **LR‑SC** lateral racing (successive halving + overflow cap), **width‑aware bar with repeat‑to‑confirm**, **short‑circuit promotion**, and **plateau‑triggered exploit/explore phases**. These match the paper’s defaults (η∈{3,4,5}, base budget b₀∈{1,2}, micro‑probe=1, overflow cap ρ∈\[0.1,0.2], slope+curvature continuation, narrow mainlines).&#x20;
* Pins seeds and budgets from `config/benchmarks.yaml` and `config/budgets.yaml`.
* Materializes:

  * `results/artifact.parquet` (single file, schema documented below).
  * `results/figures/*` and `results/figures.pdf`.
  * `results/summary.json` for CI and paper scripts.

If you only want a quick smoke test:

```bash
snakemake --profile profiles/arc fast_smoke
```

---

## 4) Monitoring & recovering runs

### 4.1 Slurm view

```bash
squeue -u $USER -o "%.18i %.10T %.10M %.20P %R %j"
tail -n 200 -f logs/slurm/*.out
```

### 4.2 Snakemake view

```bash
snakemake --profile profiles/arc --summary         # what’s done vs. pending
snakemake --profile profiles/arc -npr all          # dry-run plan with reasons
snakemake --profile profiles/arc --list-params-changes
```

### 4.3 Progress helpers

```bash
python tools/progress.py results/artifact.parquet \
  --by benchmark --show promotions,rungs,first_hit
```

### 4.4 Resume / re‑run only missing

```bash
snakemake --profile profiles/arc -R all
```

> The controller implements **freeze–thaw** of laterals across phases and short‑circuits back to exploitation on first promotions, as in the manuscript; restarting mid‑way preserves cache and rung state.&#x20;

---

## 5) Results artifact schema (for ChatGPT + LaTeX)

**`results/artifact.parquet`** (single file)

* **keys**: `benchmark`, `item_id`, `model_id`, `seed`, `budget_cap`
* **metrics**: `success@1|pass@1`, `tokens_total`, `expansions_total`, `time_to_first_hit`, `false_promotions`
* **controller** (per item): `n0_laterals`, `rungs`, `plateau_events`, `promotions`, `bars/heavy_tail_type`
* **envelopes**: `V(h)` series (smoothed), slope/curvature stats, confirmation outcomes
* **traces**: first‑hit chain (for paper figures), plus optional “fixed‑point check” results used in sect. *Mechanistic/Fixed‑point ablations* (off by default to save compute).&#x20;

The plotting rules read only this Parquet; the LaTeX builder (`snakemake paper`) consumes `results/summary.json` and `results/figures/`.

---

## 6) Committing results to the Git repo (with LFS)

> **Why LFS?** Figures and the single Parquet can exceed regular Git comfort. We track them in **Git LFS**; small summaries remain plain‑Git.

### 6.1 One‑time LFS init

```bash
git lfs install
git lfs track "results/*.parquet"
git lfs track "results/figures/*.pdf"
git add .gitattributes
git commit -m "chore: enable Git LFS for results"
```

### 6.2 Regular commit/push

```bash
git add results/artifact.parquet results/summary.json results/figures/*
git add logs/run_meta.json          # tiny, useful run context
git commit -m "results: full LToT sweep (arc), seeds/budget per config"
git push origin <your-branch>
```

> **Tip:** If ARC blocks pushes from compute nodes, do this from the login node, or clone the repo locally and `rsync` `results/` down, then commit from your laptop.

---

## 7) Tuning (only if needed)

* **Budget/width:** `config/budgets.yaml` (initial lateral width `n0`, per‑task token caps).
* **Controller params:** `config/controller.yaml` (η, ρ, micro‑probe, bar type: sub‑Gaussian / sub‑Gamma / sub‑Weibull, confirmation ON). Defaults mirror **§4.6 Design Choices and Defaults**.&#x20;
* **Mainlines:** beam/quota caps to keep depth linear; plateau thresholds (`τ`) for exploit↔explore.&#x20;
* **Models:** symbolic names → local HF paths in `config/models.yaml`.

---

## 8) Troubleshooting

**No internet on nodes**
Set:

```bash
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
```

Make sure model dirs exist under `$SCRATCH/models/*`. Use `huggingface-cli download` on a login node, or `rsync` from your laptop.

**GPU scheduling errors**
Check partitions and GPU types:

```bash
sinfo -o "%P %G %f"
```

Edit `profiles/arc/cluster.yaml` resource lines for the affected rules, e.g. `--gres=gpu:A100:1` → `gpu:L40:1`.

**Disk quota / “No space left”**
Point caches to scratch:

```bash
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
```

**Snakemake profile not found**
Run from repo root: `snakemake --profile profiles/arc all`. Use absolute paths in CI.

---

## 9) Appendix: Minimal Slurm template (for ad‑hoc tests)

If you ever need a raw batch script (you don’t for Snakemake), here’s a small template to adapt:

```bash
#!/bin/bash
#SBATCH -J ltot-test
#SBATCH -p <your-gpu-partition>
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm/%x-%j.out

module load cuda  # if ARC uses modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ltot

export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

python tools/sanity_check.py --model-path $SCRATCH/models/llama3-8b-instruct
```

---

## 10) References inside this repo

* **LToT manuscript**: controller loop, LR‑SC, width‑aware bars, promotion discipline, and defaults (see **§4.2–§4.6**). These settings are what your pipeline uses.&#x20;
* **Fixed Point Explainability**: optional ablation utilities use its “certificate up‑to‑infinity” notion for a stability check mode (disabled by default to keep cost down).&#x20;
* **Postdoc collaborator (Oxford)**: background and publications (LM benchmarking, explainability, robustness) per the CV you supplied.&#x20;

---

### Quick checklist (copy/paste)

* [ ] `ssh` to ARC, `git clone` repo.
* [ ] Create `$SCRATCH/{models,data,results,logs}` and symlink from repo.
* [ ] `conda env create -f envs/ltot.yaml && conda activate ltot`.
* [ ] `huggingface-cli login` (if needed) and **download models to `$SCRATCH/models/*`**.
* [ ] `export HF_HOME=$SCRATCH/.cache/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`.
* [ ] **Run**: `snakemake --profile profiles/arc all`.
* [ ] **Monitor**: `squeue -u $USER`, `tail -f logs/slurm/*.out`, `snakemake --summary`.
* [ ] **Commit results** (with Git LFS): add `results/artifact.parquet`, `results/figures/*`, `results/summary.json`.

---

**That’s it.** This guide assumes no prior ARC experience and keeps you offline‑safe, reproducible, and paper‑ready. If you want me to include your actual ARC partition/QoS names in `profiles/arc/cluster.yaml`, paste the output of `sinfo`/`sacctmgr` and I’ll slot them into the profile so everything “just works” with the single command.
