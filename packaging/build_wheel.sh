#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE=wheel
setup_env 0.7.0
setup_wheel_python

pip_install numpy pyyaml future

if [[ "$OSTYPE" == "msys" ]]; then
    # Apparently, there are some errors trying to compile ninja locally on
    # Windows, better to use conda on this case
    conda install ninja -y
else
    pip_install ninja
fi

setup_pip_pytorch_version
python setup.py clean

# Copy binaries to be included in the wheel distribution
if [[ "$(uname)" == Darwin || "$OSTYPE" == "msys" ]]; then
    python_exec="$(which python)"
    bin_path=$(dirname $python_exec)
    env_path=$(dirname $bin_path)
    if [[ "$(uname)" == Darwin ]]; then
        # Include LibPNG
        cp "$env_path/lib/libpng16.dylib" torchvision
    else
        # Include libPNG
        cp "$bin_path/Library/lib/libpng.lib" torchvision
    fi
else
    cp "/usr/lib64/libpng.so" torchvision
fi

if [[ "$OSTYPE" == "msys" ]]; then
    IS_WHEEL=1 "$script_dir/windows/internal/vc_env_helper.bat" python setup.py bdist_wheel
else
    IS_WHEEL=1 python setup.py bdist_wheel
fi
