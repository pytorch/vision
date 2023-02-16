#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE=conda
setup_env
export SOURCE_ROOT_DIR="$PWD"
setup_conda_pytorch_constraint
setup_conda_cudatoolkit_constraint
setup_visual_studio_constraint
setup_junit_results_folder
export CUDATOOLKIT_CHANNEL="nvidia"

if [[ "$PYTHON_VERSION" == "3.11" ]]; then
  export CONDA_CHANNEL_FLAGS="${CONDA_CHANNEL_FLAGS} -c malfet"
fi

conda build -c $CUDATOOLKIT_CHANNEL $CONDA_CHANNEL_FLAGS --no-anaconda-upload --no-test --python "$PYTHON_VERSION" packaging/torchvision
