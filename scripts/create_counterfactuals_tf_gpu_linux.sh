#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="counterfactuals_tf_gpu"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${REPO_ROOT}/.conda_envs/${ENV_NAME}"
CACHE_ROOT="${REPO_ROOT}/.cache"
PIP_CACHE_DIR="${REPO_ROOT}/.pip_cache"
TMPDIR="${REPO_ROOT}/.tmp"

mkdir -p "${REPO_ROOT}/.conda_envs" "${CACHE_ROOT}" "${PIP_CACHE_DIR}" "${TMPDIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found on PATH." >&2
  exit 1
fi

echo "Creating environment at: ${ENV_PREFIX}"
conda create --prefix "${ENV_PREFIX}" python=3.8.18 -y

echo "Installing Conda CUDA runtime used by the legacy TensorFlow stack..."
conda install --prefix "${ENV_PREFIX}" -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y

echo "Upgrading pip..."
conda run --prefix "${ENV_PREFIX}" python -m pip install --upgrade pip

echo "Installing TensorFlow stack..."
conda run --prefix "${ENV_PREFIX}" env \
  XDG_CACHE_HOME="${CACHE_ROOT}" \
  PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
  TMPDIR="${TMPDIR}" \
  python -m pip install \
  "numpy==1.24.3" \
  "tensorflow==2.10.1" \
  "keras==2.10.0" \
  "protobuf==3.19.6" \
  "h5py==3.7.0"

echo "Installing counterfactual experiment dependencies..."
conda run --prefix "${ENV_PREFIX}" env \
  XDG_CACHE_HOME="${CACHE_ROOT}" \
  PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
  TMPDIR="${TMPDIR}" \
  python -m pip install \
  "alibi==0.9.4" \
  "dask==2023.5.0" \
  "distributed==2023.5.0" \
  "fastdtw==0.3.4" \
  "matplotlib==3.7.3" \
  "matrixprofile==1.1.10" \
  "numba==0.58.0" \
  "pandas==2.0.3" \
  "plotly==5.24.1" \
  "pytest==8.3.5" \
  "requests==2.31.0" \
  "scikit-learn==1.3.0" \
  "scikit-optimize==0.9.0" \
  "scipy==1.10.1" \
  "seaborn==0.13.0" \
  "shap==0.42.1" \
  "stumpy==1.13.0" \
  "torchsummary==1.5.1" \
  "tqdm==4.65.2" \
  "TSInterpret==0.4.6" \
  "tslearn==0.6.2" \
  "wildboar==1.1.4"

echo "Installing CUDA-enabled PyTorch..."
conda run --prefix "${ENV_PREFIX}" env \
  XDG_CACHE_HOME="${CACHE_ROOT}" \
  PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
  TMPDIR="${TMPDIR}" \
  python -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.4.1+cu124"

echo "Verifying TensorFlow and PyTorch..."
conda run --prefix "${ENV_PREFIX}" python -c \
  "import tensorflow as tf; import torch; print('tf', tf.__version__, tf.config.list_physical_devices('GPU')); print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

echo
echo "Done."
echo "Activate with:"
echo "  conda activate ${ENV_PREFIX}"
