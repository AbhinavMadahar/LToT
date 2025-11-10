#!/usr/bin/env bash
set -euo pipefail
export LTOT_MODEL=${LTOT_MODEL:-sshleifer/tiny-gpt2}
export LTOT_BUDGET=${LTOT_BUDGET:-96}
export LTOT_SEEDS="${LTOT_SEEDS:-1}"
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
bash "$(dirname "$0")/run.sh"
