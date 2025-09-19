
from __future__ import annotations
import os, json, argparse, sys
from pathlib import Path
from ltot.harness import run_all

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--artifact", required=True)
    args = p.parse_args()

    try:
        import yaml
    except ImportError:
        print("PyYAML missing; attempting to install in-place is not supported in batch. Please ensure env has PyYAML.", file=sys.stderr)
        raise

    with open(args.config) as yf:
        cfg = yaml.safe_load(yf)

    run_all(cfg, args.artifact)

if __name__ == "__main__":
    main()
