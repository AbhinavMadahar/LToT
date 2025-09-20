#!/usr/bin/env bash
# ARC entry: warm cache (optional), submit 100-shard array, then aggregate.
set -euo pipefail
mode="${1:-full}"

# Common env (recommended for persistent caches)
export HF_HOME="${HF_HOME:-${DATA:-$HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

if [[ "$mode" == "full" ]]; then
  sbatch scripts/arc_warmup.sbatch || true
  jid=$(sbatch --array=0-99 scripts/arc_shard.sbatch | awk '{print $4}')
  sbatch --dependency=afterok:$jid scripts/arc_aggregate.sbatch
elif [[ "$mode" == "core" ]]; then
  # Core-first: focus on key tasks/models; remove these envs to run the rest later
  export LTOT_TASKS="gsm_hard,math_500,humaneval,mbpp_lite,game24"
  export LTOT_MODELS="llama-3.1-8b-instruct,mixtral-8x7b-instruct"
  export LTOT_SEEDS="1,2,3"
  sbatch scripts/arc_warmup.sbatch || true
  jid=$(sbatch --array=0-99 scripts/arc_shard.sbatch | awk '{print $4}')
  sbatch --dependency=afterok:$jid scripts/arc_aggregate.sbatch
else
  echo "Usage: bash launch_arc.sh [full|core]"
  exit 1
fi
