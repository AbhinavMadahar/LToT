#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then set -a; source .env; set +a; fi

# Snakemake submission on Slurm
conda run -n ltot-exp snakemake -j 200 --profile profiles/arc-slurm
