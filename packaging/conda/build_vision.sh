#!/usr/bin/env bash
if [[ -x "/remote/anaconda_token" ]]; then
    . /remote/anaconda_token || true
fi

set -ex

if [[ "$CIRCLECI" == 'true' ]]; then
    export PATH="/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin:.:$PATH"
fi

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

# Parse arguments and determmine version
###########################################################
if [[ -n "$DESIRED_CUDA" && -n "$TORCHVISION_BUILD_VERSION" && -n "$TORCHVISION_BUILD_NUMBER" ]]; then
    desired_cuda="$DESIRED_CUDA"
    build_version="$PYTORCH_BUILD_VERSION"
    build_number="$PYTORCH_BUILD_NUMBER"
else
    if [ "$#" -ne 3 ]; then
        echo "Illegal number of parameters. Pass cuda version, pytorch version, build number"
        echo "CUDA version should be Mm with no dot, e.g. '80'"
        echo "DESIRED_PYTHON should be M.m, e.g. '2.7'"
        exit 1
    fi

    desired_cuda="$1"
    build_version="$2"
    build_number="$3"
fi
if [[ "$desired_cuda" != cpu ]]; then
  desired_cuda="$(echo $desired_cuda | tr -d cuda. )"
fi
echo "Building cuda version $desired_cuda and torchvision version: $build_version build_number: $build_number"

if [[ "$desired_cuda" == 'cpu' ]]; then
    cpu_only=1
    cuver="cpu"
else
    # Switch desired_cuda to be M.m to be consistent with other scripts in
    # pytorch/builder
    export FORCE_CUDA=1
    cuda_nodot="$desired_cuda"

    if [[ ${#cuda_nodot} -eq 2 ]]; then
        desired_cuda="${desired_cuda:0:1}.${desired_cuda:1:1}"
    elif [[ ${#cuda_nodot} -eq 3 ]]; then
        desired_cuda="${desired_cuda:0:2}.${desired_cuda:2:1}"
    else
        echo "unknown cuda version $cuda_nodot"
        exit 1
    fi

    cuver="cu$cuda_nodot"
fi

export TORCHVISION_BUILD_VERSION=$build_version
export TORCHVISION_BUILD_NUMBER=$build_number

if [[ -z "$DESIRED_PYTHON" ]]; then
    DESIRED_PYTHON=('3.5' '3.6' '3.7')
fi

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

if [[ -z "$WIN_PACKAGE_WORK_DIR" ]]; then
    WIN_PACKAGE_WORK_DIR="$(echo $(pwd -W) | tr '/' '\\')\\tmp_conda_$(date +%H%M%S)"
fi

mkdir -p "$WIN_PACKAGE_WORK_DIR" || true
vision_rootdir="$(realpath ${WIN_PACKAGE_WORK_DIR})/torchvision-src"
git config --system core.longpaths true

if [[ ! -d "$vision_rootdir" ]]; then
    rm -rf "$vision_rootdir"
    git clone "https://github.com/pytorch/vision" "$vision_rootdir"
    pushd "$vision_rootdir"
    git checkout $PYTORCH_BRANCH
    popd
fi

cd "$SOURCE_DIR"

export tmp_conda="${WIN_PACKAGE_WORK_DIR}\\conda"
export miniconda_exe="${WIN_PACKAGE_WORK_DIR}\\miniconda.exe"
rm -rf "$tmp_conda"
rm -f "$miniconda_exe"
curl -sSk https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o "$miniconda_exe"
"$SOURCE_DIR/install_conda.bat" && rm "$miniconda_exe"
pushd $tmp_conda
export PATH="$(pwd):$(pwd)/Library/usr/bin:$(pwd)/Library/bin:$(pwd)/Scripts:$(pwd)/bin:$PATH"
popd
retry conda install -yq conda-build

ANACONDA_USER=pytorch-nightly
conda config --set anaconda_upload no


export TORCHVISION_PACKAGE_SUFFIX=""
if [[ "$desired_cuda" == 'cpu' ]]; then
    export CONDA_CUDATOOLKIT_CONSTRAINT=""
    export CONDA_CPUONLY_FEATURE="- cpuonly # [not osx]"
    export CUDA_VERSION="None"
else
    export CONDA_CPUONLY_FEATURE=""
    . ./switch_cuda_version.sh $desired_cuda
    if [[ "$desired_cuda" == "10.2" ]]; then
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=10.2,<10.3 # [not osx]"
    elif [[ "$desired_cuda" == "10.1" ]]; then
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=10.1,<10.2 # [not osx]"
    elif [[ "$desired_cuda" == "10.0" ]]; then
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=10.0,<10.1 # [not osx]"
    elif [[ "$desired_cuda" == "9.2" ]]; then
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=9.2,<9.3 # [not osx]"
    elif [[ "$desired_cuda" == "9.0" ]]; then
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=9.0,<9.1 # [not osx]"
    elif [[ "$desired_cuda" == "8.0" ]]; then
        export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=8.0,<8.1 # [not osx]"
    else
        echo "unhandled desired_cuda: $desired_cuda"
        exit 1
    fi
fi

if [[ -z "$PYTORCH_VERSION" ]]; then
    export CONDA_CHANNEL_FLAGS="-c pytorch-nightly"
    export PYTORCH_VERSION="$(conda search --json 'pytorch[channel=pytorch-nightly]' | \
                                python -c "import os, sys, json, re; cuver = '$cuver'; \
                                cuver = cuver.replace('cu', 'cuda') if cuver != 'cpu' else cuver; \
                                print(re.sub(r'\\+.*$', '', \
                                [x['version'] for x in json.load(sys.stdin)['pytorch'] \
                                    if (x['platform'] == 'darwin' or cuver in x['fn']) \
                                    and 'py' + os.environ['DESIRED_PYTHON'] in x['fn']][-1]))")"
    if [[ -z "$PYTORCH_VERSION" ]]; then
        echo "PyTorch version auto detection failed"
        echo "No package found for desired_cuda=$desired_cuda and DESIRED_PYTHON=$DESIRED_PYTHON"
        exit 1
    fi
else
    export CONDA_CHANNEL_FLAGS="-c pytorch -c pytorch-nightly"
fi
if [[ "$desired_cuda" == 'cpu' ]]; then
    export CONDA_PYTORCH_BUILD_CONSTRAINT="- pytorch==$PYTORCH_VERSION"
    export CONDA_PYTORCH_CONSTRAINT="- pytorch==$PYTORCH_VERSION"
else
    export CONDA_PYTORCH_BUILD_CONSTRAINT="- pytorch==${PYTORCH_VERSION}"
    export CONDA_PYTORCH_CONSTRAINT="- pytorch==${PYTORCH_VERSION}"
fi

# Loop through all Python versions to build a package for each
for py_ver in "${DESIRED_PYTHON[@]}"; do
    build_string="py${py_ver}_${build_string_suffix}"
    folder_tag="${build_string}_$(date +'%Y%m%d')"

    # Create the conda package into this temporary folder. This is so we can find
    # the package afterwards, as there's no easy way to extract the final filename
    # from conda-build
    output_folder="out_$folder_tag"
    rm -rf "$output_folder"
    mkdir "$output_folder"

    if [[ "$py_ver" == 3.5 ]]; then
	export CONDA_TYPING_CONSTRAINT="- typing"
    else
	export CONDA_TYPING_CONSTRAINT=""
    fi

    export VSTOOLCHAIN_PACKAGE=vs2017

    # We need to build the compiler activation scripts first on Windows
    time VSDEVCMD_ARGS=${VSDEVCMD_ARGS[@]} \
        conda build -c "$ANACONDA_USER" \
                    --no-anaconda-upload \
                    --output-folder "$output_folder" \
                    ../$VSTOOLCHAIN_PACKAGE

    cp ../$VSTOOLCHAIN_PACKAGE/conda_build_config.yaml ../torchvision/conda_build_config.yaml

    conda config --set anaconda_upload no
    echo "Calling conda-build at $(date)"
    if [[ "$desired_cuda" == "9.2" ]]; then
        time CMAKE_ARGS=${CMAKE_ARGS[@]} \
            BUILD_VERSION="$TORCHVISION_BUILD_VERSION" \
            CU_VERSION="$cuver" \
            SOURCE_ROOT_DIR="$vision_rootdir" \
            conda build -c "$ANACONDA_USER" \
                        -c defaults \
                        -c conda-forge \
                        -c "numba/label/dev" \
                        --no-anaconda-upload \
                        --python "$py_ver" \
                        --output-folder "$output_folder" \
                        --no-verify \
                        --no-test \
                        ../torchvision
    else
        time CMAKE_ARGS=${CMAKE_ARGS[@]} \
            BUILD_VERSION="$TORCHVISION_BUILD_VERSION" \
            CU_VERSION="$cuver" \
            SOURCE_ROOT_DIR="$vision_rootdir" \
            conda build -c "$ANACONDA_USER" \
                        -c defaults \
                        -c conda-forge \
                        --no-anaconda-upload \
                        --python "$py_ver" \
                        --output-folder "$output_folder" \
                        --no-verify \
                        --no-test \
                        ../torchvision
    fi
    echo "Finished conda-build at $(date)"

    # Extract the package for testing
    ls -lah "$output_folder"
    built_package="$(find $output_folder/ -name '*torchvision*.tar.bz2')"

    # Copy the built package to the host machine for persistence before testing
    if [[ -n "$PYTORCH_FINAL_PACKAGE_DIR" ]]; then
        mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR" || true
        cp "$built_package" "$PYTORCH_FINAL_PACKAGE_DIR/"
    fi
done


set +e
