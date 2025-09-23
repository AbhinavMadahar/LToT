#!/usr/bin/env python3
import sys, time, yaml, requests

def check(url):
    # vLLM defaults to OpenAI-compatible API. /v1/models is a simple readiness probe.
    try:
        r = requests.get(f"{url.rstrip('/')}/v1/models", timeout=3)
        ok = r.status_code == 200
        return ok, r.status_code
    except Exception as e:
        return False, str(e)

def main(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    all_ok = True
    for group, urls in cfg.get("models", {}).items():
        print(f"\n[{group}]")
        for u in urls:
            ok, detail = check(u)
            status = "OK " if ok else "BAD"
            print(f"  {status}  {u}   ({detail})")
            all_ok = all_ok and ok
    if not all_ok:
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python health_check.py endpoints.yaml")
        sys.exit(2)
    main(sys.argv[1])
