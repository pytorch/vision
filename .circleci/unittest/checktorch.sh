#!/usr/bin/env bash
set -ex

# unittest workflow for GPU may run as unittest for CPU.
# To avoid this potential issue, the workflow must stop if we find the torch.cuda.is_available is False in any event.

shopt -s nocasematch
if [[ "${CIRCLE_JOB}" == "unittest"*"gpu"* ]]; then
    torch_cuda_available=$(python -c "import torch; print(torch.cuda.is_available())")
    echo torch.cuda.is_available is $torch_cuda_available
    if [ "$torch_cuda_available" == "False" ]; then
        echo "It's unittest for GPU but torch.cuda.is_available() is False"
        exit 1
    fi
fi
shopt -u nocasematch
