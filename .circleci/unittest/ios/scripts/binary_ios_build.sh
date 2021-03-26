#!/bin/bash
set -ex -o pipefail

echo ""
echo "DIR: $(pwd)"
WORKSPACE=/Users/distiller/workspace
PROJ_ROOT_IOS=/Users/distiller/project/ios
export TCLLIBPATH="/usr/local/lib"

# install conda
curl --retry 3 -o ~/conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
chmod +x ~/conda.sh
/bin/bash ~/conda.sh -b -p ~/anaconda
export PATH="~/anaconda/bin:${PATH}"
source ~/anaconda/bin/activate

# install dependencies
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi requests typing_extensions wget --yes
conda install -c conda-forge valgrind --yes
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# sync submodules
cd ${PROJ_ROOT_IOS}
git submodule sync
git submodule update --init --recursive

# clone main pytorch repo
mkdir -p ${PROJ_ROOT_IOS}/lib
mkdir -p ${PROJ_ROOT_IOS}/build
git clone --recursive https://github.com/pytorch/pytorch.git
TORCH_ROOT="${PROJ_ROOT_IOS}/pytorch"

# run build script
chmod a+x ${TORCH_ROOT}/scripts/build_ios.sh
echo "########################################################"
cat ${TORCH_ROOT}/scripts/build_ios.sh
echo "########################################################"
echo "IOS_ARCH: ${IOS_ARCH}"
echo "IOS_PLATFORM: ${IOS_PLATFORM}"
export IOS_ARCH=${IOS_ARCH}
export IOS_PLATFORM=${IOS_PLATFORM}
unbuffer ${TORCH_ROOT}/scripts/build_ios.sh 2>&1 | ts

LIBTORCH_HEADER_ROOT="${TORCH_ROOT}/build_ios/install/include"
cd ${PROJ_ROOT_IOS}
IOS_ARCH=${IOS_ARCH} LIBTORCH_HEADER_ROOT=${LIBTORCH_HEADER_ROOT} ./build_ios.sh
rm -rf ${TORCH_ROOT}

# store the binary
DEST_DIR=${WORKSPACE}/ios/${IOS_ARCH}
mkdir -p ${DEST_DIR}
cp ${PROJ_ROOT_IOS}/lib/*.a ${DEST_DIR}
