#!/usr/bin/env bash
# Full experiment with the default matrix, three seeds, resumable.
# Re-run the same command after any interruption; finished work is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
: "${HF_TOKEN:?set HF_TOKEN (Gemma is gated)}"

latent-stats hardware
latent-stats run-all --all-seeds "$@"
echo "report: results/main/report/report.html"
