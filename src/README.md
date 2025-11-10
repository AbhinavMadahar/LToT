# Lateral Tree-of-Thoughts (LToT) — Exact Paper Replica

This repo reproduces the full experiment suite from the LToT manuscript:
- Equal-compute (±2%) parity across CoT / ToT / MCTS-PW / LToT
- Predictive continuation (slope+curvature) with width-aware bars + repeat-to-confirm
- Plateau trigger for exploitation phases
- Width-scaling sweeps (N0 ∈ {32,64,128,256,512,1024})
- Five ablations: overflow_off, no_curvature, no_width_bar, no_short_circuit, no_plateau, plus no_confirm
- Noisy-v study (LM-scored exploration) with selectivity / false-promotion logging
- Robustness toggles (heavy-tail & correlation noise) for Sec. 5.3
- Early-stop latency logging (separate pass)
- Cost-law logging (per-rung costs/expansions) for N log N fit & rung CV
- Positioning diagnostics (SH-only lateralization; SH-on-mainlines)
- Canonical dataset lists; fail-closed unless overridden

Outputs a single artifact (`results/ltot_artifact.jsonl`) that includes all tables/metrics and a main SVG figure.

---

## Quick start (Local diagnostic)

A tiny, CPU-friendly diagnostic that exercises CoT/ToT/MCTS-PW/LToT end-to-end, no ARC needed:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
export HF_HOME="$PWD/.cache/huggingface"; export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"; mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
python scripts/diag_local.py
python -m ltot.run aggregate \
  --inputs results/raw --inputs_width results/raw_width --inputs_ablate results/raw_ablate --inputs_latency results/raw_latency \
  --artifact results/ltot_artifact.jsonl --fig figures/main_equal_compute.svg


## Option‑A quick start (GSM‑Plus, equal‑compute)

Use the provided `configs/experiments_optionA.yaml` (single 8B model, GSM‑Plus, pinned 250‑ID split) and run:

```bash
# Equal‑compute run (writes Success@1 with exact‑match)
python -m ltot.run run --model llama-3.1-8b-instruct --task gsm_plus --budget 700 --seed 1 --out results/raw/optA.eq.jsonl

# Early‑stop pass (expansions‑to‑first‑verified)
python -m ltot.run earlystop --model llama-3.1-8b-instruct --task gsm_plus --budget 700 --seed 1 --out results/raw_latency/optA.lat.jsonl

# Optional width scaling at fixed compute (N0 ∈ {64,128})
python -m ltot.run widthscale --model llama-3.1-8b-instruct --task gsm_plus --budget 700 --seed 1 --out results/raw_width/optA.ws.jsonl

# Aggregate: writes main figure + metrics, including median_expansions_to_first_verified
python -m ltot.run aggregate --inputs results/raw --inputs_width results/raw_width --inputs_ablate results/raw_ablate \
    --inputs_latency results/raw_latency --artifact results/ltot_artifact.jsonl --fig figures/main_equal_compute.svg
```
