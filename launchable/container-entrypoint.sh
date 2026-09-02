#!/usr/bin/env bash
set -euo pipefail

cd /workspace/launchable
export PYTHONPATH="/workspace/launchable/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export MEGATRON_BRIDGE_DIR="/workspace/storage/Megatron-Bridge"

BRIDGE_COMMIT="8bc33cd2ca1cd044e1520130f4c5e1f5e0181434"
MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
MODEL_REVISION="b3caaabed0263651a17dc1f2d4ce97e794f76c44"

retry() {
  local attempt=1
  local maximum=4
  local delay=5
  until "$@"; do
    if (( attempt >= maximum )); then
      echo "Command failed after ${maximum} attempts: $*" >&2
      return 1
    fi
    sleep "${delay}"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}

echo "[container 1/6] Verifying PyTorch supplied by the pinned NeMo image"
python - <<'PY'
import torch

print(f"Preinstalled PyTorch: {torch.__version__}")
print(f"CUDA runtime visible to PyTorch: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise RuntimeError("The NeMo container's preinstalled PyTorch cannot initialize CUDA.")
PY

echo "[container 2/6] Installing lightweight notebook dependencies"
retry python -m pip install --disable-pip-version-check -r requirements-lab.txt
python -m pip install --disable-pip-version-check --no-deps --editable .

echo "[container 3/6] Resolving the pinned Megatron-Bridge source"
if [ ! -d "${MEGATRON_BRIDGE_DIR}/.git" ]; then
  git clone --filter=blob:none --recurse-submodules \
    https://github.com/NVIDIA-NeMo/Megatron-Bridge.git "${MEGATRON_BRIDGE_DIR}"
fi
git -C "${MEGATRON_BRIDGE_DIR}" fetch --depth 1 origin "${BRIDGE_COMMIT}"
git -C "${MEGATRON_BRIDGE_DIR}" checkout --detach "${BRIDGE_COMMIT}"
git -C "${MEGATRON_BRIDGE_DIR}" submodule update --init --recursive --depth 1
export PYTHONPATH="${MEGATRON_BRIDGE_DIR}/src:${MEGATRON_BRIDGE_DIR}/3rdparty/Megatron-LM:${PYTHONPATH}"

python - <<'PY'
from megatron.bridge.recipes.nemotronh import (
    nemotron_3_5_lightning_peft_config,
    nemotron_3_5_lightning_sft_config,
)
import vllm

assert callable(nemotron_3_5_lightning_peft_config)
assert callable(nemotron_3_5_lightning_sft_config)
print("Megatron-Bridge Lightning PEFT and full-SFT recipes are importable.")
print(f"vLLM local evaluation backend: {vllm.__version__}")
PY

echo "[container 4/6] Optional model prefetch"
if [ "${NEMOTRON_PREFETCH_MODEL:-0}" = "1" ]; then
  export PREFETCH_MODEL_ID="${MODEL_ID}"
  export PREFETCH_MODEL_REVISION="${MODEL_REVISION}"
  python - <<'PY'
import os

from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["PREFETCH_MODEL_ID"],
    revision=os.environ["PREFETCH_MODEL_REVISION"],
)
print(f"Model snapshot ready at {path}")
PY
fi

echo "[container 5/6] Optional reusable checkpoint conversion"
if [ "${NEMOTRON_PREFETCH_MODEL:-0}" = "1" ]; then
  python scripts/convert_checkpoint.py \
    --hf-model "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --output /workspace/storage/checkpoints/lightning35-megatron
fi

echo "[container 6/6] Starting JupyterLab"
exec jupyter lab \
  --allow-root \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --ServerApp.root_dir=/workspace/launchable \
  --ServerApp.default_url=/lab/tree/notebooks/01_cloud_api_baseline.ipynb \
  --ServerApp.allow_origin='*' \
  --ServerApp.token='' \
  --ServerApp.password=''
