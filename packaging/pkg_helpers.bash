# A set of useful bash functions for common functionality we need to do in
# many build scripts

# Respecting PYTHON_VERSION and UNICODE_ABI, add (or install) the correct
# version of Python to perform a build.  Relevant to wheel builds.
setup_python() {
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

# Fill CUDA_SUFFIX and CU_VERSION with CUDA_VERSION. CUDA_SUFFIX is
# left blank for the default CUDA version (that's a blank suffix)
setup_cuda_suffix() {
  if [[ "$(uname)" == Darwin ]]; then
    if [[ "$CUDA_VERSION" != "cpu" ]]; then
      echo "CUDA_VERSION on OS X must be cpu"
      exit 1
    fi
    export CPU_SUFFIX=""
    export CU_VERSION="cpu"
  else
    case "$CUDA_VERSION" in
      10.0)
        export CUDA_SUFFIX=""
        export CU_VERSION="cu100"
        ;;
      9.2)
        export CUDA_SUFFIX="+cu92"
        export CU_VERSION="cu92"
        ;;
      cpu)
        export CUDA_SUFFIX="+cpu"
        export CU_VERSION="cpu"
        ;;
      *)
        echo "Unrecognized CUDA_VERSION=$CUDA_VERSION"
    esac
    export CPU_SUFFIX="+cpu"
  fi
}

# If a package is cpu-only, we never provide a cuda suffix
setup_cpuonly_cuda_suffix() {
  export CUDA_SUFFIX=""
  export CPU_SUFFIX=""
}

# Fill BUILD_VERSION and BUILD_NUMBER if it doesn't exist already with a nightly string
# Usage: setup_build_version 0.2
setup_build_version() {
  if [[ -z "$BUILD_VERSION" ]]; then
    export BUILD_VERSION="$1.dev$(date "+%Y%m%d")"
  fi
}

# Set some useful variables for OS X, if applicable
setup_macos() {
  if [[ "$(uname)" == Darwin ]]; then
    export MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++
  fi
}

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
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
    pip_install --pre torch -f "https://download.pytorch.org/whl/nightly/${CU_VERSION}/torch_nightly.html"
    if [[ "$CUDA_VERSION" == "cpu" ]]; then
      # CUDA and CPU are ABI compatible on the CPU-only parts, so strip
      # in this case
      export PYTORCH_VERSION="$(pip show torch | grep ^Version: | sed 's/Version:  *//' | sed 's/+.\+//')"
    else
      export PYTORCH_VERSION="$(pip show torch | grep ^Version: | sed 's/Version:  *//')"
    fi
  else
    # TODO: Maybe add staging too
    pip_install "torch==$PYTORCH_VERSION$CUDA_SUFFIX" \
      -f https://download.pytorch.org/whl/torch_stable.html
  fi
}

# Fill PYTORCH_VERSION with the latest conda nightly version, and
# CONDA_CHANNEL_FLAGS with appropriate flags to retrieve these versions
#
# You MUST have populated CUDA_SUFFIX before hand.
#
# TODO: This is currently hard-coded for CPU-only case
setup_conda_pytorch_constraint() {
  if [[ -z "$PYTORCH_VERSION" ]]; then
    export CONDA_CHANNEL_FLAGS="-c pytorch-nightly"
    export PYTORCH_VERSION="$(conda search --json 'pytorch[channel=pytorch-nightly]' | python -c "import sys, json, re; print(re.sub(r'\\+.*$', '', json.load(sys.stdin)['pytorch'][-1]['version']))")"
  else
    export CONDA_CHANNEL_FLAGS="-c pytorch"
  fi
  if [[ "$CUDA_VERSION" == cpu ]]; then
    export CONDA_PYTORCH_BUILD_CONSTRAINT="- pytorch==$PYTORCH_VERSION${CPU_SUFFIX}"
    export CONDA_PYTORCH_CONSTRAINT="- pytorch==$PYTORCH_VERSION"
  else
    export CONDA_PYTORCH_BUILD_CONSTRAINT="- pytorch==${PYTORCH_VERSION}${CUDA_SUFFIX}"
    export CONDA_PYTORCH_CONSTRAINT="- pytorch==${PYTORCH_VERSION}${CUDA_SUFFIX}"
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
