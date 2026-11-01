#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f vendor/ns-3.44/CMakeLists.txt ]]; then
  echo "vendor/ns-3.44 is missing. Use the packaged source archive or run:"
  echo "git clone --branch ns-3.44 --depth 1 https://github.com/nsnam/ns-3-dev-git.git vendor/ns-3.44"
  exit 1
fi

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/build_ns3.py
.venv/bin/python -m pytest -q tests

echo
echo "Setup complete. Edit configs/mac_cmdp_experiment.yaml, then run:"
echo ".venv/bin/python scripts/run_mac_cmdp_experiment.py train --output /data/4t/lwk/results/YOUR_RUN"
