#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -q -r requirements-colab.txt
if ! python -c 'import torch' >/dev/null 2>&1; then
  python -m pip install -q 'torch>=2.6,<3'
fi
if ! command -v Rscript >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends r-base r-base-dev
fi
Rscript scripts/install_r.R
