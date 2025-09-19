
# LToT Experiments + Paper Build (ARC-ready)

**This file is documentation intended for codebase maintainers. If you are interested in running the experiments, please read `quickstart.md` instead.**

This bundle adds a **single-command** pipeline to run the LToT experiments from the manuscript and to build your LaTeX PDFs. It is designed for **Oxford ARC** (Slurm) and **Hugging Face local models** (offline-friendly). All experiment outputs are persisted to **one artifact file** (`results/artifact.jsonl`) that ChatGPT (or any script) can read to update the TeX automatically.

> **TL;DR (first run on ARC):**
>
> ```bash
> # 1) copy bundle into repo and extract
> tar -xzf ltot_bundle.tar.gz
>
> # 2) one-time env setup (ARC)
> bash scripts/setup_arc_env.sh    # creates Conda env: ltot-exp
>
> # 3) run all experiments + build PDFs with Slurm profile
> bash scripts/submit_all.sh
>
> # 4) check progress / logs
> bash scripts/check_status.sh
>
> # 5) when done, commit results (artifact + tables)
> bash scripts/commit_results.sh "Run on ARC on $(date +%F)"
> ```

## What’s inside

- **Snakemake workflow (merged)**: Includes your paper build Snakefile and a new experiments workflow. Top-level `Snakefile` orchestrates both.
- **ARC/Slurm profile**: `profiles/arc-slurm/config.yaml` (uses Snakemake’s `executor: slurm`), with sensible defaults for **A100/H100**, and fallbacks for **L40/L4/V100**.
- **LToT controller (full)**: Implements LR–SC racing with width-aware bars, **repeat-to-confirm**, **short-circuit promotion**, **dual-score frontier**, and **plateau detection** for mainlines.
- **HF local-model harness**: Loads models with `transformers` from a local path or cache; **no outbound network required**. Credentials are read from environment if needed.
- **Verifiers**: Exact-match for math, unit tests for code, and a conservative dual gate for QA (plausibility + path consistency).
- **Single-file artifact**: `results/artifact.jsonl` contains *all* results and figures (PNGs as base64) with stable IDs so a TeX-updater (or ChatGPT) can patch references.
- **Helper scripts**: One-command submit, status, and git commit.

## Oxford ARC quickstart

If this is the first time on ARC for this project:

```bash
module purge
bash scripts/setup_arc_env.sh
```

The script will:
- Load recommended ARC modules (Anaconda, GCC, CUDA when available).
- Create/activate a Conda env `ltot-exp` with PyTorch, Transformers, Datasets, Snakemake.
- Set `HF_HOME` and `TRANSFORMERS_OFFLINE=1` by default (you can flip it in `.env`).

Then launch everything (experiments + LaTeX builds) with:

```bash
bash scripts/submit_all.sh
```

This runs Snakemake with the **Slurm executor**, spreading jobs over GPUs you request in `config/experiments.yaml` (per-task GPU counts & constraints).

### Secrets & credentials

- Copy `.env.sample` to `.env` and fill if needed. The workflow will export only the necessary variables to job environments.
- Hugging Face: the harness prefers local model paths; if you need auth to access a private model repo, set `HF_TOKEN` in `.env`. No tokens are written to disk beyond `.env`.

## Outputs

- `results/artifact.jsonl` — **single file** with every metric, table, and figure (images are base64 PNG blobs). Schema is detailed in `docs/ARTIFACT.md`.
- `results/tables/*.csv` — machine-friendly tables mirrored from the artifact (optional aids for human inspection).
- Your LaTeX PDFs as defined in your paper Snakefile.

## One-command re-run

Change budgets/models in `config/*.yaml`, then:

```bash
bash scripts/submit_all.sh
```

## Support

If Snakemake or Slurm versions differ on ARC, adjust `profiles/arc-slurm/config.yaml` accordingly (Snakemake 8+ is assumed).
