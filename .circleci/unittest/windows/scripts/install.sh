#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -ex

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
conda activate ./env

echo $CU_VERSION

if [ "${CU_VERSION:-}" == "cpu" ] ; then
    cudatoolkit="cpuonly"
else
    # only support cuda major version >= 10
    CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION"
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
    cudatoolkit="cudatoolkit=${version}"
fi

printf "Installing PyTorch with %s\n" "${cudatoolkit}"
conda install -y -c "pytorch-${UPLOAD_CHANNEL}" -c conda-forge "pytorch-${UPLOAD_CHANNEL}"::pytorch "${cudatoolkit}" pytest

if [ $PYTHON_VERSION == "3.6" ]; then
    printf "Installing minimal PILLOW version\n"
    # Install the minimal PILLOW version. Otherwise, let setup.py install the latest
    pip install pillow>=5.3.0
fi

if [ "$cudatoolkit" != "cpuonly" ]; then
    # check torch 
    python -c "import torch; print(torch.cuda.is_available())"
    # check cuda driver version
    for path in '/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe' /c/Windows/System32/nvidia-smi.exe; do
        if [[ -x "$path" ]]; then
            "$path" || echo "true";
            break
        fi
    done

    # set the cuda env vars
    source "$this_dir/set_cuda_envs.sh"
fi

printf "* Installing torchvision\n"
"$this_dir/vc_env_helper.bat" python setup.py develop
