@echo on

set TORCHVISION_BUILD_VERSION=%PKG_VERSION%
set TORCHVISION_BUILD_NUMBER=%PKG_BUILDNUM%

if not "%CUDA_VERSION%" == "None" (
    set build_with_cuda=1
    set desired_cuda=%CUDA_VERSION:~0,-1%.%CUDA_VERSION:~-1,1%
) else (
    set build_with_cuda=
)

if "%build_with_cuda%" == "" goto cuda_flags_end

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%desired_cuda%
set CUDA_BIN_PATH=%CUDA_PATH%\bin
set NVCC_FLAGS=-D__CUDA_NO_HALF_OPERATORS__ --expt-relaxed-constexpr
if "%desired_cuda%" == "9.0" set NVCC_FLAGS=%NVCC_FLAGS% -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_50,code=compute_50
if "%desired_cuda%" == "10.0" set NVCC_FLAGS=%NVCC_FLAGS% -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_50,code=compute_50

:cuda_flags_end

python setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit /b 1
