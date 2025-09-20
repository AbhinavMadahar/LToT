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
