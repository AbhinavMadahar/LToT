#!/usr/bin/env bash
set -euo pipefail
msg="${1:-'Update results'}"
git add results/artifact.jsonl results/tables/*.csv 2>/dev/null || true
git add results 2>/dev/null || true
git commit -m "$msg" || echo "No changes to commit."
