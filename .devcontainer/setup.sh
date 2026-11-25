#!/usr/bin/env bash
# Runs once when the Codespace is created: install WSBench (CPU torch) and download the model and ARC data.
set -euo pipefail
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.4'
python -m pip install -e . 'pytest>=8'
# The app also works without these downloads (tiny model only), so a failed download must not block the Codespace.
wsbench fetch all || echo "wsbench fetch failed; run 'wsbench fetch all' again later"
