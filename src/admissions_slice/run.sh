#!/usr/bin/env bash
set -euo pipefail

# Resolve paths
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_SRC"

# Defaults (overridable via env)
MODEL="${LTOT_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
BUDGET="${LTOT_BUDGET:-1400}"
SEEDS="${LTOT_SEEDS:-1 2}"
TASK="math_500"

# Offline-friendly cache (override HF_HOME to reuse caches between runs)
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HOME="${HF_HOME:-$REPO_SRC/.hf_cache}"
mkdir -p "$HF_HOME"

# Results layout
RAW="results/raw"
RAW_W="results/raw_width"
RAW_A="results/raw_ablate"
RAW_L="results/raw_latency"
OUTDIR="results/admissions_slice"
mkdir -p "$RAW" "$RAW_W" "$RAW_A" "$RAW_L" "$OUTDIR"

# Swap in admissions config
CFG_DIR="configs"
ORIG_CFG="$CFG_DIR/experiments.yaml"
AS_CFG_REL="admissions_slice/configs/experiments.yaml"
AS_CFG="$AS_CFG_REL"
BACKUP="configs/experiments.yaml.bak.$(date +%s)"
if [[ -f "$ORIG_CFG" ]]; then
  cp "$ORIG_CFG" "$BACKUP"
fi
cp "$AS_CFG" "$ORIG_CFG"

# Trap to restore config on exit
restore_cfg() {
  if [[ -f "$BACKUP" ]]; then
    mv -f "$BACKUP" "$ORIG_CFG"
  fi
}
trap restore_cfg EXIT

echo "[admissions_slice] MODEL=$MODEL BUDGET=$BUDGET TASK=$TASK SEEDS=($SEEDS)"
python -V || true

# Main runs
for S in $SEEDS; do
  python -m ltot.run run        --model "$MODEL" --task "$TASK" --budget "$BUDGET" --seed "$S" --out "$RAW/run_seed${S}.jsonl" --shard 0
  # Small width scaling (N0 controlled inside config; this will log by width choices in config)
  python -m ltot.run widthscale --model "$MODEL" --task "$TASK" --budget "$BUDGET" --seed "$S" --out "$RAW_W/width_seed${S}.jsonl" --shard 0
  # Early-stop latency / expansions-to-first-verified
  python -m ltot.run earlystop  --model "$MODEL" --task "$TASK" --budget "$BUDGET" --seed "$S" --out "$RAW_L/latency_seed${S}.jsonl" --shard 0
done

# Aggregate into single artifact+figure
python -m ltot.run aggregate \
  --inputs "$RAW" \
  --inputs_width "$RAW_W" \
  --inputs_ablate "$RAW_A" \
  --inputs_latency "$RAW_L" \
  --artifact "$OUTDIR/artifacts.jsonl" \
  --fig "$OUTDIR/eq_compute.svg"

echo "[admissions_slice] Completed. Artifacts in $OUTDIR"
