#!/usr/bin/env bash
set -euo pipefail

# 可覆盖：COVRA_CONDA_ENV_NAME、COVRA_PROJECT_ROOT、COVRA_PYTHON
ENV_NAME="${COVRA_CONDA_ENV_NAME:-dico-rank}"
PROJECT_ROOT="${COVRA_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -n "${COVRA_PYTHON:-}" ]]; then
  PYTHON_BIN="${COVRA_PYTHON}"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" python=3.10
  fi
  conda activate "${ENV_NAME}"
  PYTHON_BIN="$(command -v python)"
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel packaging ninja
"${PYTHON_BIN}" -m pip install -r requirements.txt

"${PYTHON_BIN}" - <<'PY'
import torch
import transformers
import accelerate
import datasets

print("python environment ready")
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("datasets", datasets.__version__)
print("attention_backend", "torch_sdpa")
PY
