if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Pass cuda version"
    echo "CUDA version should be cu90, cu100 or cpu"
    exit 1
fi
export CUVER="$1" # cu90 cu100 cpu

export TORCHVISION_BUILD_VERSION="0.3.0"
export TORCHVISION_BUILD_NUMBER="1"
export OUT_DIR="/remote/$CUVER"
export TORCH_WHEEL="torch -f https://download.pytorch.org/whl/$CUVER/stable.html --no-index"

pushd /opt/python
DESIRED_PYTHON=(*/)
popd
for desired_py in "${DESIRED_PYTHON[@]}"; do
    python_installations+=("/opt/python/$desired_py")
done

OLD_PATH=$PATH
git clone https://github.com/pytorch/vision -b v${TORCHVISION_BUILD_VERSION}
pushd vision
for PYDIR in "${python_installations[@]}"; do
    export PATH=$PYDIR/bin:$OLD_PATH
    pip install numpy pyyaml future
    pip install $TORCH_WHEEL
    pip install ninja
    python setup.py clean
    python setup.py bdist_wheel
    mkdir -p $OUT_DIR
    cp dist/*.whl $OUT_DIR/
done
