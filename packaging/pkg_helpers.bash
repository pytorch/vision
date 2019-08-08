# A set of useful bash functions for common functionality we need to do in
# many build scripts


# Setup CUDA environment variables, based on CU_VERSION
#
# Inputs:
#   CU_VERSION (cpu, cu92, cu100)
#   NO_CUDA_PACKAGE (bool)
#
# Outputs:
#   VERSION_SUFFIX (e.g., "")
#   PYTORCH_VERSION_SUFFIX (e.g., +cpu)
#   WHEEL_DIR (e.g., cu100/)
#   CUDA_HOME (e.g., /usr/local/cuda-9.2, respected by torch.utils.cpp_extension)
#   FORCE_CUDA (respected by torchvision setup.py)
#   NVCC_FLAGS (respected by torchvision setup.py)
#
# Precondition: CUDA versions are installed in their conventional locations in
# /usr/local/cuda-*
#
# NOTE: Why VERSION_SUFFIX versus PYTORCH_VERSION_SUFFIX?  If you're building
# a package with CUDA on a platform we support CUDA on, VERSION_SUFFIX ==
# PYTORCH_VERSION_SUFFIX and everyone is happy.  However, if you are building a
# package with only CPU bits (e.g., torchaudio), then VERSION_SUFFIX is always
# empty, but PYTORCH_VERSION_SUFFIX is +cpu (because that's how you get a CPU
# version of a Python package.  But that doesn't apply if you're on OS X,
# since the default CU_VERSION on OS X is cpu.
setup_cuda() {
  if [[ "$(uname)" == Darwin ]] || [[ -n "$NO_CUDA_PACKAGE" ]]; then
    if [[ "$CU_VERSION" != "cpu" ]]; then
      echo "CU_VERSION on OS X / package with no CUDA must be cpu"
      exit 1
    fi
    if [[ "$(uname)" == Darwin ]]; then
      export PYTORCH_VERSION_SUFFIX=""
    else
      export PYTORCH_VERSION_SUFFIX="+cpu"
    fi
    export VERSION_SUFFIX=""
    # NB: When there is no CUDA package available, we put these
    # packages in the top-level directory, so they are eligible
    # for selection even if you are otherwise trying to install
    # a cu100 stack.  This differs from when there ARE CUDA packages
    # available; then we don't want the cpu package; we want
    # to give you as much goodies as possible.
    export WHEEL_DIR=""
  else
    case "$CU_VERSION" in
      cu100)
        export PYTORCH_VERSION_SUFFIX=""
        export CUDA_HOME=/usr/local/cuda-10.0/
        export FORCE_CUDA=1
        # Hard-coding gencode flags is temporary situation until
        # https://github.com/pytorch/pytorch/pull/23408 lands
        export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_50,code=compute_50"
        ;;
      cu92)
        export CUDA_HOME=/usr/local/cuda-9.2/
        export PYTORCH_VERSION_SUFFIX="+cu92"
        export FORCE_CUDA=1
        export NVCC_FLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_50,code=compute_50"
        ;;
      cpu)
        export PYTORCH_VERSION_SUFFIX="+cpu"
        ;;
      *)
        echo "Unrecognized CU_VERSION=$CU_VERSION"
    esac
    export VERSION_SUFFIX="$PYTORCH_VERSION_SUFFIX"
    export WHEEL_DIR="$CU_VERSION/"
  fi
}

# Populate build version if necessary, and add version suffix
#
# Inputs:
#   BUILD_VERSION (e.g., 0.2.0 or empty)
#
# Outputs:
#   BUILD_VERSION (e.g., 0.2.0.dev20190807+cpu)
#
# Fill BUILD_VERSION if it doesn't exist already with a nightly string
# Usage: setup_build_version 0.2.0
setup_build_version() {
  if [[ -z "$BUILD_VERSION" ]]; then
    export BUILD_VERSION="$1.dev$(date "+%Y%m%d")$VERSION_SUFFIX"
  else
    export BUILD_VERSION="$BUILD_VERSION$VERSION_SUFFIX"
  fi
}

# Set some useful variables for OS X, if applicable
setup_macos() {
  if [[ "$(uname)" == Darwin ]]; then
    export MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++
  fi
}

# Top-level entry point for things every package will need to do
#
# Usage: setup_env 0.2.0
setup_env() {
  setup_cuda
  setup_build_version "$1"
  setup_macos
}

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

# Inputs:
#   PYTHON_VERSION (2.7, 3.5, 3.6, 3.7)
#   UNICODE_ABI (bool)
#
# Outputs:
#   PATH modified to put correct Python version in PATH
#
# Precondition: If Linux, you are in a soumith/manylinux-cuda* Docker image
setup_wheel_python() {
  if [[ "$(uname)" == Darwin ]]; then
    eval "$(conda shell.bash hook)"
    conda env remove -n "env$PYTHON_VERSION" || true
    conda create -yn "env$PYTHON_VERSION" python="$PYTHON_VERSION"
    conda activate "env$PYTHON_VERSION"
  else
    case "$PYTHON_VERSION" in
      2.7)
        if [[ -n "$UNICODE_ABI" ]]; then
          python_abi=cp27-cp27mu
        else
          python_abi=cp27-cp27m
        fi
        ;;
      3.5) python_abi=cp35-cp35m ;;
      3.6) python_abi=cp36-cp36m ;;
      3.7) python_abi=cp37-cp37m ;;
      *)
        echo "Unrecognized PYTHON_VERSION=$PYTHON_VERSION"
        exit 1
        ;;
    esac
    export PATH="/opt/python/$python_abi/bin:$PATH"
  fi
}

# Install with pip a bit more robustly than the default
pip_install() {
  retry pip install --progress-bar off "$@"
}

# Install torch with pip, respecting PYTORCH_VERSION, and record the installed
# version into PYTORCH_VERSION, if applicable
setup_pip_pytorch_version() {
  if [[ -z "$PYTORCH_VERSION" ]]; then
    # Install latest prerelease version of torch, per our nightlies, consistent
    # with the requested cuda version
    pip_install --pre torch -f "https://download.pytorch.org/whl/nightly/${WHEEL_DIR}torch_nightly.html"
    if [[ "$CUDA_VERSION" == "cpu" ]]; then
      # CUDA and CPU are ABI compatible on the CPU-only parts, so strip
      # in this case
      export PYTORCH_VERSION="$(pip show torch | grep ^Version: | sed 's/Version:  *//' | sed 's/+.\+//')"
    else
      export PYTORCH_VERSION="$(pip show torch | grep ^Version: | sed 's/Version:  *//')"
    fi
  else
    pip_install "torch==$PYTORCH_VERSION$CUDA_SUFFIX" \
      -f https://download.pytorch.org/whl/torch_stable.html \
      -f https://download.pytorch.org/whl/nightly/torch_nightly.html
  fi
}

# Fill PYTORCH_VERSION with the latest conda nightly version, and
# CONDA_CHANNEL_FLAGS with appropriate flags to retrieve these versions
#
# You MUST have populated CUDA_SUFFIX before hand.
setup_conda_pytorch_constraint() {
  if [[ -z "$PYTORCH_VERSION" ]]; then
    export CONDA_CHANNEL_FLAGS="-c pytorch-nightly"
    export PYTORCH_VERSION="$(conda search --json 'pytorch[channel=pytorch-nightly]' | python -c "import sys, json, re; print(re.sub(r'\\+.*$', '', json.load(sys.stdin)['pytorch'][-1]['version']))")"
  else
    export CONDA_CHANNEL_FLAGS="-c pytorch -c pytorch-nightly"
  fi
  if [[ "$CUDA_VERSION" == cpu ]]; then
    export CONDA_PYTORCH_BUILD_CONSTRAINT="- pytorch==$PYTORCH_VERSION${PYTORCH_VERSION_SUFFIX}"
    export CONDA_PYTORCH_CONSTRAINT="- pytorch==$PYTORCH_VERSION"
  else
    export CONDA_PYTORCH_BUILD_CONSTRAINT="- pytorch==${PYTORCH_VERSION}${PYTORCH_VERSION_SUFFIX}"
    export CONDA_PYTORCH_CONSTRAINT="- pytorch==${PYTORCH_VERSION}${PYTORCH_VERSION_SUFFIX}"
  fi
}

# Translate CUDA_VERSION into CUDA_CUDATOOLKIT_CONSTRAINT
setup_conda_cudatoolkit_constraint() {
  export CONDA_CPUONLY_FEATURE=""
  if [[ "$(uname)" == Darwin ]]; then
    export CONDA_CUDATOOLKIT_CONSTRAINT=""
  else
    case "$CUDA_VERSION" in
      10.0)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=10.0,<10.1 # [not osx]"
        ;;
      9.2)
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=9.2,<9.3 # [not osx]"
        ;;
      cpu)
        export CONDA_CUDATOOLKIT_CONSTRAINT=""
        export CONDA_CPUONLY_FEATURE="- cpuonly"
        ;;
    esac
  fi
}
