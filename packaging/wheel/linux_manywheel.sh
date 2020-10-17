#!/bin/bash
set -ex

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Pass cuda version"
    echo "CUDA version should be cu92, cu100 or cpu"
    exit 1
fi
export CUVER="$1" # cu[0-9]* cpu

if [[ "$CUVER" == "cu102" ]]; then
  cu_suffix=""
else
  cu_suffix="+$CUVER"
fi

export TORCHVISION_BUILD_VERSION="0.4.0.dev$(date "+%Y%m%d")${cu_suffix}"
export TORCHVISION_BUILD_NUMBER="1"
export TORCHVISION_LOCAL_VERSION_LABEL="$CUVER"
export OUT_DIR="/remote/$CUVER"

pushd /opt/python
DESIRED_PYTHON=(*/)
popd
for desired_py in "${DESIRED_PYTHON[@]}"; do
    python_installations+=("/opt/python/$desired_py")
done

OLD_PATH=$PATH
cd /tmp
rm -rf vision
git clone https://github.com/pytorch/vision

cd /tmp/vision

for PYDIR in "${python_installations[@]}"; do
    export PATH=$PYDIR/bin:$OLD_PATH
    pip install --upgrade pip
    pip install numpy pyyaml future

    pip uninstall -y torch || true
    pip uninstall -y torch_nightly || true

    export TORCHVISION_PYTORCH_DEPENDENCY_NAME=torch_nightly
    pip install torch_nightly -f https://download.pytorch.org/whl/nightly/$CUVER/torch_nightly.html
    # CPU/CUDA variants of PyTorch have ABI compatible PyTorch for
    # the CPU only bits.  Therefore, we
    # strip off the local package qualifier, but ONLY if we're
    # doing a CPU build.
    if [[ "$CUVER" == "cpu" ]]; then
        export TORCHVISION_PYTORCH_DEPENDENCY_VERSION="$(pip show torch_nightly | grep ^Version: | sed 's/Version: \+//' | sed 's/+.\+//')"
    else
        export TORCHVISION_PYTORCH_DEPENDENCY_VERSION="$(pip show torch_nightly | grep ^Version: | sed 's/Version: \+//')"
    fi
    echo "Building against ${TORCHVISION_PYTORCH_DEPENDENCY_VERSION}"

    pip install ninja
    python setup.py clean
    python setup.py bdist_wheel
    mkdir -p $OUT_DIR
    cp dist/*.whl $OUT_DIR/
done
