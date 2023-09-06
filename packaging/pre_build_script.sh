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
  # Install libpng from Anaconda (defaults)
  conda install libpng -yq
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

  # Install native CentOS libJPEG, freetype and GnuTLS
  yum install -y libjpeg-turbo-devel freetype gnutls

  # Download all the dependencies required to compile image and video_reader
  # extensions
  mkdir -p ext_libraries
  pushd ext_libraries
  popd
  export PATH="$(pwd)/ext_libraries/bin:$PATH"
  pip install auditwheel

  # Point to custom libraries
  export LD_LIBRARY_PATH=$(pwd)/ext_libraries/lib:$LD_LIBRARY_PATH
  export TORCHVISION_INCLUDE=$(pwd)/ext_libraries/include
  export TORCHVISION_LIBRARY=$(pwd)/ext_libraries/lib
fi

pip install numpy pyyaml future ninja
pip install --upgrade setuptools
