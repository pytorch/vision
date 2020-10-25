#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE=wheel
setup_env 0.8.0
setup_wheel_python
pip_install numpy pyyaml future ninja
setup_pip_pytorch_version
python setup.py clean

# Copy binaries to be included in the wheel distribution
if [[ "$(uname)" == Darwin || "$OSTYPE" == "msys" ]]; then
    python_exec="$(which python)"
    bin_path=$(dirname $python_exec)
    env_path=$(dirname $bin_path)
    if [[ "$(uname)" == Darwin ]]; then
        # Install delocate to relocate the required binaries
        pip_install delocate
    else
        cp "$bin_path/Library/bin/libpng16.dll" torchvision
        cp "$bin_path/Library/bin/libjpeg.dll" torchvision
    fi
else
    # Install auditwheel to get some inspection utilities
    pip_install auditwheel

    # Point to custom libraries
    export LD_LIBRARY_PATH=$(pwd)/ext_libraries/lib:$LD_LIBRARY_PATH
    export TORCHVISION_INCLUDE=$(pwd)/ext_libraries/include
    export TORCHVISION_LIBRARY=$(pwd)/ext_libraries/lib
fi

download_copy_ffmpeg

if [[ "$OSTYPE" == "msys" ]]; then
    IS_WHEEL=1 "$script_dir/windows/internal/vc_env_helper.bat" python setup.py bdist_wheel
else
    IS_WHEEL=1 python setup.py bdist_wheel
fi


if [[ "$(uname)" == Darwin ]]; then
    pushd dist/
    python_exec="$(which python)"
    bin_path=$(dirname $python_exec)
    env_path=$(dirname $bin_path)
    for whl in *.whl; do
        DYLD_LIBRARY_PATH="$env_path/lib/:$DYLD_LIBRARY_PATH" delocate-wheel -v $whl
    done
else
    if [[ "$OSTYPE" == "msys" ]]; then
        "$script_dir/windows/internal/vc_env_helper.bat" python $script_dir/wheel/relocate.py
    else
        LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH" python $script_dir/wheel/relocate.py
    fi
fi
