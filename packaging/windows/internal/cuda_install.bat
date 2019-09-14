@echo on

if "%CUDA_VERSION%" == "cpu" (
    echo Skipping for CPU builds
    exit /b 0
)

set SRC_DIR=%~dp0\..

if not exist "%SRC_DIR%\temp_build" mkdir "%SRC_DIR%\temp_build"

set /a CUDA_VER=%CUDA_VERSION%
set CUDA_VER_MAJOR=%CUDA_VERSION:~0,-1%
set CUDA_VER_MINOR=%CUDA_VERSION:~-1,1%
set CUDA_VERSION_STR=%CUDA_VER_MAJOR%.%CUDA_VER_MINOR%

IF %CUDA_VER% EQU 92 goto cuda92
IF %CUDA_VER% EQU 100 goto cuda100

echo CUDA %CUDA_VERSION_STR% is not supported
exit /b 1

:cuda92
IF NOT EXIST "%SRC_DIR%\temp_build\cuda_9.2.148_win10.exe" (
    curl -k -L https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers2/cuda_9.2.148_win10 --output "%SRC_DIR%\temp_build\cuda_9.2.148_win10.exe"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\cuda_9.2.148_win10.exe"
    set "ARGS=nvcc_9.2 cuobjdump_9.2 nvprune_9.2 cupti_9.2 visual_studio_integration_9.2 cublas_9.2 cublas_dev_9.2 cudart_9.2 cufft_9.2 cufft_dev_9.2 curand_9.2 curand_dev_9.2 cusolver_9.2 cusolver_dev_9.2 cusparse_9.2 cusparse_dev_9.2 nvgraph_9.2 nvgraph_dev_9.2 npp_9.2 npp_dev_9.2 nvrtc_9.2 nvrtc_dev_9.2 nvml_dev_9.2"
)

IF NOT EXIST "%SRC_DIR%\temp_build\cudnn-9.2-windows10-x64-v7.2.1.38.zip" (
    curl -k -L https://downloads.sourceforge.net/project/cuda-dnn/7/CUDA-9.2/cudnn-9.2-windows10-x64-v7.2.1.38.zip --output "%SRC_DIR%\temp_build\cudnn-9.2-windows10-x64-v7.2.1.38.zip"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\cudnn-9.2-windows10-x64-v7.2.1.38.zip"
)

goto cuda_common

:cuda100

IF NOT EXIST "%SRC_DIR%\temp_build\cuda_10.0.130_411.31_win10.exe" (
    curl -k -L https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_411.31_win10 --output "%SRC_DIR%\temp_build\cuda_10.0.130_411.31_win10.exe"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\cuda_10.0.130_411.31_win10.exe"
    set "ARGS=nvcc_10.0 cuobjdump_10.0 nvprune_10.0 cupti_10.0 visual_studio_integration_10.0 cublas_10.0 cublas_dev_10.0 cudart_10.0 cufft_10.0 cufft_dev_10.0 curand_10.0 curand_dev_10.0 cusolver_10.0 cusolver_dev_10.0 cusparse_10.0 cusparse_dev_10.0 nvgraph_10.0 nvgraph_dev_10.0 npp_10.0 npp_dev_10.0 nvrtc_10.0 nvrtc_dev_10.0 nvml_dev_10.0"
)

IF NOT EXIST "%SRC_DIR%\temp_build\cudnn-10.0-windows10-x64-v7.4.1.5.zip" (
    curl -k -L https://www.dropbox.com/s/9v1z9rmbjw9mhx2/cudnn-10.0-windows10-x64-v7.4.1.5.zip?dl=1 --output "%SRC_DIR%\temp_build\cudnn-10.0-windows10-x64-v7.4.1.5.zip"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\cudnn-10.0-windows10-x64-v7.4.1.5.zip"
)

goto cuda_common

:cuda_common

echo Installing CUDA toolkit...

start /wait "%CUDA_SETUP_FILE%" -s %ARGS%

if not exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%" (
    echo CUDA %CUDA_VERSION_STR% installed failed.
)

echo Installing cuDNN...
7z x %CUDNN_SETUP_FILE% -o"%SRC_DIR%\temp_build\cudnn"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\cuda\bin\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\cuda\lib\x64\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\lib\x64"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\cuda\include\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\include"

echo Cleaning temp files
rd /s /q "%SRC_DIR%\temp_build" || ver > nul
pushd "C:\NVIDIA"
rd /s /q .  || ver > nul
popd
