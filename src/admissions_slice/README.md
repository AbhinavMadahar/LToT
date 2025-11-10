# LToT — Admissions Slice (Push‑Button)

This folder provides a **push‑button** pipeline to reproduce the admissions‑subset for **LToT** on **MATH‑500 (mini)** with:
- equal‑compute comparisons (CoT / ToT / **LToT**),
- a **tiny width** sweep at fixed budget, and
- **expansions‑to‑first‑verified** (early‑stop latency).

## Quickstart (bare metal)
From the project `src/` directory:
```bash
bash admissions_slice/run.sh
```

Environment overrides:
```bash
LTOT_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0" LTOT_BUDGET=1400 LTOT_SEEDS="1 2" bash admissions_slice/run.sh
```

Outputs land in `results/admissions_slice/`:
- `artifacts.jsonl` — unified artifact stream (metrics and events)
- `eq_compute.svg`  — small summary figure
- plus per‑run JSONL in `results/raw*`

## Slurm (one GPU)
```bash
sbatch admissions_slice/run_slurm.sbatch
```

## Dataset subset
This slice uses a **deterministic mini split** for MATH‑500 via
```
admissions_slice/data/math500_mini_ids.txt
```
Adjust IDs if you prefer a different subset. The config is at:
```
admissions_slice/configs/experiments.yaml
```
and is copied over `configs/experiments.yaml` during the run.

## Offline / Repro
Set `TRANSFORMERS_OFFLINE=1` (default here) and ensure you have the needed model weights in `HF_HOME`. The run script sets `HF_HOME` to `./.hf_cache` by default.

## Notes
- The main LToT controller knobs come from the config here; change them if you want a different width or thresholds.
- For admissions, keep seeds small (2–3) and budget modest; full preregistered sweeps belong in January.
