#!/bin/bash

if [[ "$(uname)" == Darwin ]]; then
  # Uninstall Conflicting jpeg brew formulae
  jpeg_packages=$(brew list | grep jpeg)
  echo "Existing Jpeg-related Brew libraries"
  echo $jpeg_packages
  for pkg in $jpeg_packages; do
    brew uninstall --ignore-dependencies --force $pkg || true
  done

  conda install -yq wget
fi

if [[ "$(uname)" == Darwin || "$OSTYPE" == "msys" ]]; then
  conda install libpng libwebp -yq
  # Installing webp also installs a non-turbo jpeg, so we uninstall jpeg stuff
  # before re-installing them
  conda uninstall libjpeg-turbo libjpeg -y
  conda install -yq ffmpeg=4.2 libjpeg-turbo -c pytorch

  # Copy binaries to be included in the wheel distribution
  if [[ "$OSTYPE" == "msys" ]]; then
      python_exec="$(which python)"
      bin_path=$(dirname $python_exec)
      cp "$bin_path/Library/bin/libjpeg.dll" torchvision
  fi
else

  if [[ "$ARCH" == "aarch64" ]]; then
    conda install libpng -yq
    conda install -yq ffmpeg=4.2 libjpeg-turbo -c pytorch-nightly
  fi

  yum install -y libjpeg-turbo-devel libwebp-devel freetype gnutls
  pip install auditwheel
fi

pip install numpy pyyaml future ninja
pip install --upgrade setuptools==72.1.0
