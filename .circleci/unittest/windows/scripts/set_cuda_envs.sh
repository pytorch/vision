#!/usr/bin/env bash

if [ "${CU_VERSION:-}" != "cpu" ] ; then
    if [[ ${#CU_VERSION} -eq 4 ]]; then
        CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
    elif [[ ${#CU_VERSION} -eq 5 ]]; then
        CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi
    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION"
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"

    # set cuda envs
    export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${version}/bin:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${version}/libnvvp:$PATH"
    export CUDA_PATH_V${version/./_}="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${version}"
    export CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v${version}"
    export CUDA_TOOLKIT_ROOT_DIR="$CUDA_PATH"
    export CUDAToolkit_ROOT="$CUDA_PATH"

   if  [ ! -d "$CUDA_PATH" ]
   then
        echo "$CUDA_PATH" does not exist
        exit 1
   fi
fi
