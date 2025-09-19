
import os
import json
import subprocess

configfile: "config/experiments.yaml"

RUN_ARTIFACT = "results/artifact.jsonl"

rule run_experiments:
    output:
        artifact=RUN_ARTIFACT
    conda:
        "workflow/envs/ltot-exp.yml"
    resources:
        gpus=config['slurm_defaults'].get('gpus', 1),
        cpus=config['slurm_defaults'].get('cpus', 8),
        mem_mb=config['slurm_defaults'].get('mem_mb', 32000),
        runtime=config['slurm_defaults'].get('runtime', "04:00:00"),
        partition=config['slurm_defaults'].get('partition', "gpu"),
        qos=config['slurm_defaults'].get('qos', "standard"),
        account=config['slurm_defaults'].get('account', ""),
        constraint=config['slurm_defaults'].get('constraint', "")
    envmodules:
        "anaconda"
    shell:
        """
        set -euo pipefail
        export PYTHONUNBUFFERED=1
        # export env from .env if present
        if [ -f .env ]; then set -a; source .env; set +a; fi
        python -u scripts/run_experiments.py \
            --config config/experiments.yaml \
            --artifact {output.artifact}
        """
