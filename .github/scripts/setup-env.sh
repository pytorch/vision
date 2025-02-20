#!/usr/bin/env bash

set -euxo pipefail

# Prepare conda
set +x && eval "$($(which conda) shell.bash hook)" && set -x

# Setup the OS_TYPE environment variable that should be used for conditions involving the OS below.
case $(uname) in
  Linux)
    OS_TYPE=linux
    ;;
  Darwin)
    OS_TYPE=macos
    ;;
  MSYS*)
    OS_TYPE=windows
    ;;
  *)
    echo "Unknown OS type:" $(uname)
    exit 1
    ;;
esac

echo '::group::Create build environment'
# See https://github.com/pytorch/vision/issues/7296 for ffmpeg
conda create \
  --name ci \
  --quiet --yes \
  python="${PYTHON_VERSION}" pip \
  ninja cmake \
  libpng \
  libwebp \
  'ffmpeg<4.3'
conda activate ci
conda install --quiet --yes libjpeg-turbo -c pytorch
pip install --progress-bar=off --upgrade setuptools==72.1.0

# See https://github.com/pytorch/vision/issues/6790
if [[ "${PYTHON_VERSION}" != "3.11" ]]; then
  pip install --progress-bar=off av!=10.0.0
fi

echo '::endgroup::'

if [[ "${OS_TYPE}" == windows && "${GPU_ARCH_TYPE}" == cuda ]]; then
  echo '::group::Install VisualStudio CUDA extensions on Windows'
  TARGET_DIR="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/MSBuild/Microsoft/VC/v170/BuildCustomizations"
  mkdir -p "${TARGET_DIR}"
  cp -r "${CUDA_HOME}/MSBuildExtensions/"* "${TARGET_DIR}"
  echo '::endgroup::'
fi

echo '::group::Install PyTorch'
# TODO: Can we maybe have this as environment variable in the job template? For example, `IS_RELEASE`.
if [[ (${GITHUB_EVENT_NAME} = 'pull_request' && (${GITHUB_BASE_REF} = 'release'*)) || (${GITHUB_REF} = 'refs/heads/release'*) ]]; then
  CHANNEL=test
else
  CHANNEL=nightly
fi

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
PYTORCH_WHEEL_INDEX="https://download.pytorch.org/whl/${CHANNEL}/${GPU_ARCH_ID}"
pip install --progress-bar=off --pre torch --index-url="${PYTORCH_WHEEL_INDEX}"

if [[ $GPU_ARCH_TYPE == 'cuda' ]]; then
  python -c "import torch; exit(not torch.cuda.is_available())"
fi
echo '::endgroup::'

echo '::group::Install third party dependencies prior to TorchVision install'
# Installing with `easy_install`, e.g. `python setup.py install` or `python setup.py develop`, has some quirks when
# when pulling in third-party dependencies. For example:
# - On Windows, we often hit an SSL error although `pip` can install just fine.
# - It happily pulls in pre-releases, which can lead to more problems down the line.
#   `pip` does not unless explicitly told to do so.
# Thus, we use `easy_install` to extract the third-party dependencies here and install them upfront with `pip`.
python setup.py egg_info
# The requires.txt cannot be used with `pip install -r` directly. The requirements are listed at the top and the
# optional dependencies come in non-standard syntax after a blank line. Thus, we just extract the header.
sed -e '/^$/,$d' *.egg-info/requires.txt | tee requirements.txt
pip install --progress-bar=off -r requirements.txt
echo '::endgroup::'

echo '::group::Install TorchVision'
python setup.py develop
echo '::endgroup::'

echo '::group::Install torchvision-extra-decoders'
# This can be done after torchvision was built
pip install torchvision-extra-decoders
echo '::endgroup::'

echo '::group::Collect environment information'
conda list
python -m torch.utils.collect_env
echo '::endgroup::'
