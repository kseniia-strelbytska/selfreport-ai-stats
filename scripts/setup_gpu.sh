#!/usr/bin/env bash
# One-time environment setup on the GPU machine (tested target: RTX 5090, Blackwell sm_120).
# Usage: bash scripts/setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 .venv
source .venv/bin/activate

# Blackwell (sm_120) needs CUDA 12.8+ wheels.  Adjust the index for your driver:
#   cu128 -> driver >= 570.   See https://pytorch.org/get-started/locally/
uv pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[gpu]"

# Optional fast generation backend.  vLLM wheels lag new CUDA/GPU releases;
# only install if a wheel matching your torch/CUDA exists:
#   uv pip install -e ".[vllm]"

# Gated generator (Gemma): accept the licence on huggingface.co, then
#   export HF_TOKEN=hf_xxx
# If downloads hang, disable the xet transfer backend:
#   export HF_HUB_DISABLE_XET=1

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
latent-stats hardware
