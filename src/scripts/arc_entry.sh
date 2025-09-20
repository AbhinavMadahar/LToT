#!/usr/bin/env bash
# Convenience wrapper identical to launch_arc.sh, kept under scripts/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$HERE"
bash launch_arc.sh "${1:-full}"
