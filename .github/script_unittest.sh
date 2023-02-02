#!/usr/bin/env bash

set -eu

echo '::group::Prepare conda'
BASH_CONFIG=$(conda init bash | grep modified | tr -s ' ' | cut -d ' ' -f2)
source "${BASH_CONFIG}"
echo '::endgroup::'

echo '::group::Set PyTorch conda channel'
# TODO: Can we maybe have this as environment variable in the job template? For example, `IS_RELEASE`.
if [[ (${GITHUB_EVENT_NAME} = 'pull_request' && (${GITHUB_BASE_REF} = 'release'*)) || (${GITHUB_REF} = 'refs/heads/release'*) ]]; then
  POSTFIX=test
else
  POSTFIX=nightly
fi
PYTORCH_CHANNEL=pytorch-"${POSTFIX}"
echo "${PYTORCH_CHANNEL}"
echo '::endgroup::'

echo '::group::Set PyTorch conda mutex'
case $GPU_ARCH_TYPE in
  cpu)
    PYTORCH_MUTEX=cpuonly
    ;;
  cuda)
    PYTORCH_MUTEX="pytorch-cuda=${GPU_ARCH_VERSION}"
    ;;
  *)
    echo "Unknown GPU_ARCH_TYPE=${GPU_ARCH_TYPE}"
    exit 1
    ;;
esac
echo "${PYTORCH_MUTEX}"
echo '::endgroup::'

echo '::group::Create build environment'
conda create \
  --name ci \
  --quiet --yes \
  python="${PYTHON_VERSION}" pip \
  setuptools ninja \
  libpng jpeg \
  Pillow numpy requests
conda activate ci
pip install 'av<10'
echo '::endgroup::'

echo '::group::Install PyTorch'
conda install \
  --quiet --yes \
  -c "${PYTORCH_CHANNEL}" \
  -c nvidia \
  pytorch \
  "${PYTORCH_MUTEX}"

if [[ $GPU_ARCH_TYPE = 'cuda' ]]; then
  python3 -c "import torch; exit(not torch.cuda.is_available())"
fi
echo '::endgroup::'

echo '::group::Install TorchVision'
# The `setuptools` package installed through `conda` includes a patch that errors if something is installed
# through `setuptools` while the `CONDA_BUILD` environment variable is set.
# https://github.com/AnacondaRecipes/setuptools-feedstock/blob/f5d8d256810ce28fc0cf34170bc34e06d3754041/recipe/patches/0002-disable-downloads-inside-conda-build.patch
# (Although we are not using the `-c conda-forge` channel, the patch is equivalent but not public for
# `setuptools` from the `-c defaults` channel
# TODO: investigate where `CONDA_BUILD` is set and maybe fix it there
unset CONDA_BUILD
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
