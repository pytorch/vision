#!/bin/bash
set -eux -o pipefail

# whoops, In binary_common cu_version(cu102) is used, CUDA_VERSION is used in unittest
if [[ -v CU_VERSION ]]; then
    version=${CU_VERSION/cu/}
    # from 102 -> 10.2
    CUDA_VERSION=${version%?}.${version: -1}
fi

CUDA_VERSION=11.1
cuda_major_version=${CUDA_VERSION%.*}

# cuda_installer_name
case "$CUDA_VERSION" in
    10.1 )
        cuda_installer_name="cuda_10.1.243_426.00_win10"
        ;;
    10.2 )
        cuda_installer_name="cuda_10.2.89_441.22_win10"
        ;;
    11.1 )
        cuda_installer_name="cuda_11.1.0_456.43_win10"
        ;;
    11.2 )
        cuda_installer_name="cuda_11.2.2_461.33_win10"
        ;;
    11.3 )
        cuda_installer_name="cuda_11.3.0_465.89_win10"
        ;;
    * )
        echo "CUDA_VERSION $CUDA_VERSION is not supported yet"
        exit 1
        ;;
esac

# msbuild_project_dir
case "$cuda_major_version" in
    10 )
        msbuild_project_dir="10:CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions"
        ;;
    11 )
        msbuild_project_dir="11:visual_studio_integration/CUDAVisualStudioIntegration/extras/visual_studio_integration/MSBuildExtensions"
        ;;
    * )
        echo "cuda major version $cuda_major_version isn't supported"
        exit 1
        ;;
esac

# cuda_install_packages
# https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#install-cuda-software
cuda10_packages_template="nvcc cuobjdump nvprune cupti cublas cublas_dev cudart cufft cufft_dev curand curand_dev cusolver cusolver_dev cusparse cusparse_dev nvgraph nvgraph_dev npp npp_dev nvrtc nvrtc_dev nvml_dev"

cuda11_packages_template="nvcc cuobjdump nvprune nvprof cupti cublas cublas_dev cudart cufft cufft_dev curand curand_dev cusolver cusolver_dev cusparse cusparse_dev npp npp_dev nvrtc nvrtc_dev nvml_dev"

case "$CUDA_VERSION" in
    10.1|10.2 )
        packages_template="${cuda10_packages_template}"
        ;;
    11.1|11.2 )
        packages_template="${cuda11_packages_template}"
        ;;
    11.3 )
        packages_template="${cuda11_packages_template} thrust"
        ;;
    * )
        echo "CUDA_VERSION $CUDA_VERSION isn't supported"
        exit 1
        ;;
esac

read -ra package_array <<< "$packages_template"
package_array=("${package_array[@]/%/_$CUDA_VERSION}") # add version suffix for each package
cuda_install_packages="${package_array[*]}"

if [[ "$cuda_major_version" == "11" && "${JOB_EXECUTOR}" == "windows-with-nvidia-gpu" ]]; then
    cuda_install_packages="${cuda_install_packages} Display.Driver"
fi

cuda_installer_link="https://ossci-windows.s3.amazonaws.com/${cuda_installer_name}.exe"

curl --retry 3 -kLO $cuda_installer_link
7z x ${cuda_installer_name}.exe -o${cuda_installer_name}
cd ${cuda_installer_name}
mkdir cuda_install_logs

set +e

./setup.exe -s ${cuda_install_packages} -loglevel:6 -log:"$(pwd -W)/cuda_install_logs"

set -e

# cp -r ${msbuild_project_dir}/* "C:/Program Files (x86)/Microsoft Visual Studio/2019/${VC_PRODUCT}/MSBuild/Microsoft/VC/v160/BuildCustomizations/"

if ! ls "/c/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64/nvToolsExt64_1.dll"
then
    curl --retry 3 -kLO https://ossci-windows.s3.amazonaws.com/NvToolsExt.7z
    7z x NvToolsExt.7z -oNvToolsExt
    mkdir -p "C:/Program Files/NVIDIA Corporation/NvToolsExt"
    cp -r NvToolsExt/* "C:/Program Files/NVIDIA Corporation/NvToolsExt/"
    export NVTOOLSEXT_PATH="C:\\Program Files\\NVIDIA Corporation\\NvToolsExt\\"
fi

if ! ls "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/bin/nvcc.exe"
then
    echo "CUDA installation failed"
    mkdir -p /c/w/build-results
    7z a "c:\\w\\build-results\\cuda_install_logs.7z" cuda_install_logs
    exit 1
fi

cd ..
rm -rf ./${cuda_installer_name}
rm -f ./${cuda_installer_name}.exe