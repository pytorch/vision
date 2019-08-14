@echo off

set SRC_DIR=%~dp0\..
pushd %SRC_DIR%

set PYTHON_VERSION=%PYTHON_PREFIX:py=cp%

if "%BUILD_VISION%" == "" (
    pip install future pytest coverage hypothesis protobuf
) ELSE (
    pip install future pytest "pillow>=4.1.1" mock
)

for /F "delims=" %%i in ('where /R %SRC_DIR%\output\%CUDA_PREFIX% *%MODULE_NAME%*%PYTHON_VERSION%*.whl') do pip install "%%i"

if ERRORLEVEL 1 exit /b 1

if NOT "%BUILD_VISION%" == "" (
    echo Smoke testing imports
    python -c "import torchvision"
    if ERRORLEVEL 1 exit /b 1
    goto smoke_test_end
)

echo Smoke testing imports
python -c "import torch"
if ERRORLEVEL 1 exit /b 1

python -c "from caffe2.python import core"
if ERRORLEVEL 1 exit /b 1

echo Checking that MKL is available
python -c "import torch; exit(0 if torch.backends.mkl.is_available() else 1)"
if ERRORLEVEL 1 exit /b 1

setlocal EnableDelayedExpansion
set NVIDIA_GPU_EXISTS=0
for /F "delims=" %%i in ('wmic path win32_VideoController get name') do (
    set GPUS=%%i
    if not "x!GPUS:NVIDIA=!" == "x!GPUS!" (
        SET NVIDIA_GPU_EXISTS=1
        goto gpu_check_end
    )
)
:gpu_check_end
endlocal & set NVIDIA_GPU_EXISTS=%NVIDIA_GPU_EXISTS%

if NOT "%CUDA_PREFIX%" == "cpu" if "%NVIDIA_GPU_EXISTS%" == "1" (
    echo Checking that CUDA archs are setup correctly
    python -c "import torch; torch.randn([3,5]).cuda()"
    if ERRORLEVEL 1 exit /b 1

    echo Checking that magma is available
    python -c "import torch; torch.rand(1).cuda(); exit(0 if torch.cuda.has_magma else 1)"
    if ERRORLEVEL 1 exit /b 1

    echo Checking that CuDNN is available
    python -c "import torch; exit(0 if torch.backends.cudnn.is_available() else 1)"
    if ERRORLEVEL 1 exit /b 1
)
:smoke_test_end

echo Not running unit tests. Hopefully these problems are caught by CI
goto test_end

if "%BUILD_VISION%" == "" (
    cd pytorch\test
    python run_test.py -v
) else (
    cd vision
    pytest .
)

if ERRORLEVEL 1 exit /b 1

:test_end

popd
exit /b 0
