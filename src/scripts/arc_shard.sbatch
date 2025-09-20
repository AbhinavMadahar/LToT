#!/bin/bash
#SBATCH --job-name=ltot.shard
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/arr.%x-%A_%a.out
#SBATCH --error=logs/arr.%x-%A_%a.err
set -euo pipefail
source $PWD/.venv/bin/activate
export LTOT_SHARD=${SLURM_ARRAY_TASK_ID}
echo "[INFO] Running Snakemake for LTOT_SHARD=$LTOT_SHARD"
snakemake --snakefile Snakefile --rerun-incomplete --cores 1 --notemp
