#!/bin/bash

if [[ "$(uname)" == Darwin ]]; then
  # Uninstall Conflicting jpeg brew formulae
  jpeg_packages=$(brew list | grep jpeg)
  echo "Existing Jpeg-related Brew libraries"
  echo $jpeg_packages
  for pkg in $jpeg_packages; do
    brew uninstall --ignore-dependencies --force $pkg || true
  done

  conda install -y wget
fi

if [[ "$(uname)" == Darwin || "$OSTYPE" == "msys" ]]; then
  conda install libpng libwebp>=1.3.2 -y
  # Installing webp also installs a non-turbo jpeg, so we uninstall jpeg stuff
  # before re-installing them
  conda uninstall libjpeg-turbo libjpeg -y
  conda install -y libjpeg-turbo -c pytorch

  # Copy binaries to be included in the wheel distribution
  if [[ "$OSTYPE" == "msys" ]]; then
      python_exec="$(which python)"
      bin_path=$(dirname $python_exec)
      cp "$bin_path/Library/bin/libjpeg.dll" torchvision
  fi
else

  if [[ "$ARCH" == "aarch64" ]]; then
    conda install libpng -y
    conda install -y libjpeg-turbo -c pytorch-nightly
  fi

  conda install libwebp>=1.3.2 -y
  conda install libjpeg-turbo -c pytorch
  yum install -y freetype gnutls
  pip install "auditwheel<6.3.0"
fi

# Build the image extension without rocJPEG on ROCm. setup.py enables rocJPEG
# whenever $ROCM_HOME/include/rocjpeg/rocjpeg.h exists, which links
# torchvision/image_stable.so against librocjpeg.so.1. That library is neither
# bundled into the torchvision wheel (the repair step omits it) nor shipped in
# the torch wheel, so the extension fails to dlopen at import and every
# torchvision.io image op silently disappears -- CPU jpeg/png/webp included.
# Written to BUILD_ENV_FILE because this script runs as a subprocess; the build
# steps source that file.
if [[ "${CU_VERSION:-}" == rocm* ]]; then
  echo "Disabling rocJPEG support for ${CU_VERSION}"
  export TORCHVISION_USE_ROCJPEG=0
  if [[ -n "${BUILD_ENV_FILE:-}" ]]; then
    echo "export TORCHVISION_USE_ROCJPEG=0" >> "${BUILD_ENV_FILE}"
  fi
fi

pip install numpy pyyaml future ninja
pip install --upgrade setuptools==72.1.0
