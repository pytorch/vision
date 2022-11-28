if [[ "$(uname)" == Darwin || "$OSTYPE" == "msys" ]]; then
  # Install libpng from Anaconda (defaults)
  conda install ${CONDA_CHANNEL_FLAGS} libpng -y
  if [[ "${PYTHON_VERSION}" = "3.9" || "${PYTHON_VERSION}" = "3.10" ]]; then
    conda install ${CONDA_CHANNEL_FLAGS} "jpeg=8" -y
  else
    conda install ${CONDA_CHANNEL_FLAGS} "jpeg<=9b" -y
  fi
  conda install -yq ffmpeg=4.2 -c pytorch
  conda install -yq wget
else
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
