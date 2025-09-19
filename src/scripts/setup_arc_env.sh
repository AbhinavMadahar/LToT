#!/usr/bin/env bash
set -euo pipefail

echo "[ARC] Loading common modules (adjust if your ARC image differs)"
module purge || true
module load Anaconda3 || module load anaconda || true

echo "[ARC] Creating conda env 'ltot-exp' (once)"
if ! conda env list | grep -q "^ltot-exp"; then
  conda config --set channel_priority flexible || true
  mamba env create -f workflow/envs/ltot-exp.yml || conda env create -f workflow/envs/ltot-exp.yml
else
  echo "[ARC] Env ltot-exp already exists; updating"
  mamba env update -n ltot-exp -f workflow/envs/ltot-exp.yml || conda env update -n ltot-exp -f workflow/envs/ltot-exp.yml
fi

echo "[ARC] Done. To activate:  conda activate ltot-exp"
