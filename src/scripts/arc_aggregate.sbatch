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
