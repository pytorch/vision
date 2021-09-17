#!/usr/bin/env bash

if [ "${CU_VERSION:-}" != "cpu" ] ; then
    if [[ ${#CU_VERSION} -eq 5 ]]; then
        CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi

    # It's a log to see if CU_VERSION exists, if not, we use environment CUDA_VERSION directly
    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION $CU_VERSION"

    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"

    # set cuda envs
    export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${version}/bin:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${version}/libnvvp:$PATH"
    export CUDA_PATH_V${version/./_}="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${version}"
    export CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${version}"
    export NVTOOLSEXT_PATH="C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64"

    if  [ ! -d "$CUDA_PATH" ]
    then
        echo "$CUDA_PATH" does not exist
        exit 1
    fi
    
    if [ ! -f "${CUDA_PATH}\include\nvjpeg.h" ]
    then
        echo "nvjpeg does not exist"
        exit 1
    fi
    
    # check cuda driver version
    for path in '/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe' /c/Windows/System32/nvidia-smi.exe; do
        if [[ -x "$path" ]]; then
            "$path" || echo "true";
            break
        fi
    done
    which nvcc
    nvcc --version
    env | grep CUDA
fi
