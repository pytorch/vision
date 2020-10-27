#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE=conda
setup_env 0.8.0
export SOURCE_ROOT_DIR="$PWD"
setup_conda_pytorch_constraint
setup_conda_cudatoolkit_constraint
setup_visual_studio_constraint
setup_junit_results_folder

if [[ $(uname) == "Linux" ]]; then
    mkdir -p ext_deps
    pushd ext_deps

    # PyELFtools tarball (Not found in Anaconda defaults)
    wget https://github.com/eliben/pyelftools/archive/v0.26.tar.gz
    tar -xvzf v0.26.tar.gz
    rm -rf v0.26.tar.gz

    popd
fi

conda build $CONDA_CHANNEL_FLAGS -c defaults -c conda-forge --no-anaconda-upload --python "$PYTHON_VERSION" packaging/torchvision
