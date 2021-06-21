#!/usr/bin/env bash

set -e

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
conda activate ./env

if [ "${CU_VERSION:-}" == cpu ] ; then
else
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
fi

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
pytest --cov=torchvision --junitxml=test-results/junit.xml -v --durations 20 test/test_backbone_utils.py
