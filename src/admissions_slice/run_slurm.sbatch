#!/bin/bash
#SBATCH -J ltot_admissions
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -t 16:00:00
#SBATCH -o results/logs/%x.%j.out
#SBATCH -e results/logs/%x.%j.err

module load cuda || true

export LTOT_MODEL=${LTOT_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
export LTOT_BUDGET=${LTOT_BUDGET:-1400}
export LTOT_SEEDS="${LTOT_SEEDS:-1 2}"

bash admissions_slice/run.sh
