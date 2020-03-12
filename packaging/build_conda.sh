#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE=conda
setup_env 0.6.0
export SOURCE_ROOT_DIR="$PWD"
setup_conda_pytorch_constraint
setup_conda_cudatoolkit_constraint
setup_visual_studio_constraint
setup_junit_results_folder

# Small workaround to install onnxruntime on the machine.
# just to see if it works before cleanup
if [[ "$(uname)" == Linux ]] ; then
    conda create -n build_env python="$PYTHON_VERSION" anaconda -yq
    activate build_env
    pip install -q onnxruntime
fi

conda build $CONDA_CHANNEL_FLAGS -c defaults -c conda-forge --no-anaconda-upload --python "$PYTHON_VERSION" packaging/torchvision
