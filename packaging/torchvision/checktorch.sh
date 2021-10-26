#!/usr/bin/env bash
set -ex

torch_cuda_version=$(python -c "import torch; print(torch.version.cuda)")
echo torch.version.cuda is $torch_cuda_version

shopt -s nocasematch

# 1. FORCE_CUDA is only used in package workflow
# 2. package workflow is running in CPU machine, so torch.cuda.is_available always returns False.
#    Thus we use torch.version.cuda to check the installed PyTorch.
echo FORCE_CUDA is "$FORCE_CUDA"
if [ "${FORCE_CUDA}" == "1" ] ; then
    if [ "$torch_cuda_version" == "None" ]; then
        echo "We want to build torch vision with cuda but the installed pytorch isn't with cuda"
        exit 1
    fi
fi

# In unitest workflow, torch.cuda.is_available() must be true.
if [[ "${CIRCLE_JOB}" == "unittest"*"gpu"* ]]; then
    torch_cuda_available=$(python -c "import torch; print(torch.cuda.is_available())")
    echo torch.cuda.is_available is $torch_cuda_available
    if [ "$torch_cuda_available" == "False" ]; then
        echo "It's unittest for GPU but torch.cuda.is_available() is False"
        exit 1
    fi
fi

shopt -u nocasematch
