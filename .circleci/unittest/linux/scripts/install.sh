#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -ex

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

if [ "${CU_VERSION:-}" == cpu ] ; then
    cudatoolkit="cpuonly"
    version="cpu"
else
    if [[ ${#CU_VERSION} -eq 4 ]]; then
        CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
    elif [[ ${#CU_VERSION} -eq 5 ]]; then
        CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi
    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION: ${CU_VERSION} "
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
    cudatoolkit="pytorch-cuda=${version}"

     # make sure local cuda is set to required cuda version and not CUDA version by default
    rm -f /usr/local/cuda
    ln -s /usr/local/cuda-${version} /usr/local/cuda
fi

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac

printf "Installing PyTorch with %s\n" "${cudatoolkit}"
if [ "${os}" == "MacOSX" ]; then
    conda install -y -c "pytorch-${UPLOAD_CHANNEL}" "pytorch-${UPLOAD_CHANNEL}"::pytorch "${cudatoolkit}"
else
    conda install -y -c "pytorch-${UPLOAD_CHANNEL}" -c nvidia "pytorch-${UPLOAD_CHANNEL}"::pytorch[build="*${version}*"] "${cudatoolkit}"
fi


printf "* Installing torchvision\n"
python setup.py develop
