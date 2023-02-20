#!/usr/bin/env bash

set -euo pipefail

# Prepare conda
CONDA_PATH=$(which conda)
eval "$(${CONDA_PATH} shell.bash hook)"

echo '::group::Set PyTorch conda channel and wheel index'
# TODO: Can we maybe have this as environment variable in the job template? For example, `IS_RELEASE`.
if [[ (${GITHUB_EVENT_NAME} = 'pull_request' && (${GITHUB_BASE_REF} = 'release'*)) || (${GITHUB_REF} = 'refs/heads/release'*) ]]; then
  CHANNEL_ID=test
else
  CHANNEL_ID=nightly
fi
PYTORCH_CONDA_CHANNEL=pytorch-"${CHANNEL_ID}"
echo "PYTORCH_CONDA_CHANNEL=${PYTORCH_CONDA_CHANNEL}"

case $GPU_ARCH_TYPE in
  cpu)
    GPU_ARCH_ID="cpu"
    ;;
  cuda)
    VERSION_WITHOUT_DOT=$(echo "${GPU_ARCH_VERSION}" | sed 's/\.//')
    GPU_ARCH_ID="cu${VERSION_WITHOUT_DOT}"
    ;;
  *)
    echo "Unknown GPU_ARCH_TYPE=${GPU_ARCH_TYPE}"
    exit 1
    ;;
esac
PYTORCH_WHEEL_INDEX="https://download.pytorch.org/whl/${CHANNEL_ID}/${GPU_ARCH_ID}"
echo "PYTORCH_WHEEL_INDEX=${PYTORCH_WHEEL_INDEX}"
echo '::endgroup::'

echo '::group::Create build environment'
conda create \
  --name ci \
  --quiet --yes \
  python="${PYTHON_VERSION}" pip \
  ninja libpng jpeg ffmpeg \
  -c "${PYTORCH_CONDA_CHANNEL}" \
  -c conda-forge
conda activate ci
pip install --progress-bar=off --upgrade setuptools
echo '::endgroup::'

echo '::group::Install PyTorch'
# Due to the supply chain attack in Dec 2022 (https://pytorch.org/blog/compromised-nightly-dependency/), we host all
# third-party dependencies on Linux on our own indices and *don't* install them from PyPI.
case "$(uname -s)" in
    Linux*)
      INDEX_TYPE="index-url"
      ;;
    *)
      INDEX_TYPE="extra-index-url"
esac

pip install --progress-bar=off torch "--${INDEX_TYPE}=${PYTORCH_WHEEL_INDEX}"

if [[ $GPU_ARCH_TYPE = 'cuda' ]]; then
  python3 -c "import torch; exit(not torch.cuda.is_available())"
fi
echo '::endgroup::'

echo '::group::Install TorchVision'
python setup.py develop
echo '::endgroup::'

echo '::group::Collect PyTorch environment information'
python -m torch.utils.collect_env
echo '::endgroup::'

echo '::group::Install testing utilities'
pip install --progress-bar=off pytest pytest-mock pytest-cov
echo '::endgroup::'

echo '::group::Run tests'
pytest --durations=25
echo '::endgroup::'
