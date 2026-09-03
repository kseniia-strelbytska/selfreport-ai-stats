#!/usr/bin/env bash
# A ~15-30 minute real-model check on the GPU box before committing to the
# full run: one synthetic theme, one world, 30 documents, primary run only,
# real Gemma generator and real Qwen QLoRA trainer.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
latent-stats run-all --config configs/quick_gpu.yaml "$@"
echo "report: results/quick_gpu/report/report.html"
