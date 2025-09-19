#!/usr/bin/env bash
set -euo pipefail
echo "=== Recent Snakemake logs ==="
ls -1rt .snakemake/log/* 2>/dev/null | tail -n 20 || true
echo "=== Slurm queue (your jobs) ==="
squeue -u "${USER}" || true
