
# Oxford ARC: fast start

- Load Anaconda (`module load Anaconda3`), create env via `scripts/setup_arc_env.sh`.
- Local HF models should be placed under `/scratch/$USER/models/...` and referenced in `config/experiments.yaml`.
- Default Slurm partition/qos/account can be supplied via `.env` (see `.env.sample`).
- Snakemake profile uses `executor: slurm` and resources bound from `config/experiments.yaml`.

