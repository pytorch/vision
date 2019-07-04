@echo off

set SRC_DIR=%~dp0\..

if not exist "%SRC_DIR%\temp_build" mkdir "%SRC_DIR%\temp_build"

set /a CUDA_VER=%CUDA_VERSION%
set CUDA_VER_MAJOR=%CUDA_VERSION:~0,-1%
set CUDA_VER_MINOR=%CUDA_VERSION:~-1,1%
set CUDA_VERSION_STR=%CUDA_VER_MAJOR%.%CUDA_VER_MINOR%

IF %CUDA_VER% LEQ 90 (
    set "NVCC_PACKAGE=compiler_%CUDA_VERSION_STR%"
) ELSE (
    set "NVCC_PACKAGE=nvcc_%CUDA_VERSION_STR%"
)

IF %CUDA_VER% EQU 80 goto cuda80
IF %CUDA_VER% EQU 90 goto cuda90
IF %CUDA_VER% EQU 91 goto cuda91
IF %CUDA_VER% EQU 92 goto cuda92
IF %CUDA_VER% EQU 100 goto cuda100

echo CUDA %CUDA_VERSION_STR% is not supported
exit /b 1

:cuda80

echo CUDA 8.0 is not supported
exit /b 1

:cuda90
IF NOT EXIST "%SRC_DIR%\temp_build\cuda_9.0.176_windows.7z" (
    curl -k -L https://www.dropbox.com/s/z5b7ryz0zrimntl/cuda_9.0.176_windows.7z?dl=1 --output "%SRC_DIR%\temp_build\cuda_9.0.176_windows.7z"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\cuda_9.0.176_windows.7z"
    set "NVCC_PACKAGE=compiler_%CUDA_VERSION_STR%"
)

IF NOT EXIST "%SRC_DIR%\temp_build\cudnn-9.0-windows7-x64-v7.zip" (
    curl -k -L https://www.dropbox.com/s/6p0xyqh472nu8m1/cudnn-9.0-windows7-x64-v7.zip?dl=1 --output "%SRC_DIR%\temp_build\cudnn-9.0-windows7-x64-v7.zip"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\cudnn-9.0-windows7-x64-v7.zip"
)

goto cuda_common

:cuda91

IF NOT EXIST "%SRC_DIR%\temp_build\cuda_9.1.85_windows.7z" (
    curl -k -L https://www.dropbox.com/s/7a4sbq0dln6v7t2/cuda_9.1.85_windows.7z?dl=1 --output "%SRC_DIR%\temp_build\cuda_9.1.85_windows.7z"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\cuda_9.1.85_windows.7z"
    set "NVCC_PACKAGE=nvcc_%CUDA_VERSION_STR%"
)

IF NOT EXIST "%SRC_DIR%\temp_build\cudnn-9.1-windows7-x64-v7.zip" (
    curl -k -L https://www.dropbox.com/s/e0prhgsrbyfi4ov/cudnn-9.1-windows7-x64-v7.zip?dl=1 --output "%SRC_DIR%\temp_build\cudnn-9.1-windows7-x64-v7.zip"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\cudnn-9.1-windows7-x64-v7.zip"
)

goto cuda_common

:cuda92

echo CUDA 9.2 is not supported
exit /b 1

:cuda100

echo CUDA 10.0 is not supported
exit /b 1

:cuda_common

set "CUDA_PREFIX=cuda%CUDA_VERSION%"

IF NOT EXIST "%SRC_DIR%\temp_build\NvToolsExt.7z" (
    curl -k -L https://www.dropbox.com/s/9mcolalfdj4n979/NvToolsExt.7z?dl=1 --output "%SRC_DIR%\temp_build\NvToolsExt.7z"
    if errorlevel 1 exit /b 1
)

echo Installing CUDA toolkit...

7z x %CUDA_SETUP_FILE% -o"%SRC_DIR%\temp_build\cuda"
pushd "%SRC_DIR%\temp_build\cuda"
dir
start /wait setup.exe -s %NVCC_PACKAGE% cublas_%CUDA_VERSION_STR% cublas_dev_%CUDA_VERSION_STR% cudart_%CUDA_VERSION_STR% curand_%CUDA_VERSION_STR% curand_dev_%CUDA_VERSION_STR% cusparse_%CUDA_VERSION_STR% cusparse_dev_%CUDA_VERSION_STR% nvrtc_%CUDA_VERSION_STR% nvrtc_dev_%CUDA_VERSION_STR% cufft_%CUDA_VERSION_STR% cufft_dev_%CUDA_VERSION_STR%
popd
echo Installing VS integration...
xcopy /Y "%SRC_DIR%\temp_build\cuda\_vs\*.*" "c:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\V140\BuildCustomizations"

echo Installing NvToolsExt...
7z x %SRC_DIR%\temp_build\NvToolsExt.7z -o"%SRC_DIR%\temp_build\NvToolsExt"
mkdir "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\bin\x64"
mkdir "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\include"
mkdir "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\lib\x64"
xcopy /Y "%SRC_DIR%\temp_build\NvToolsExt\bin\x64\*.*" "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\bin\x64"
xcopy /Y "%SRC_DIR%\temp_build\NvToolsExt\include\*.*" "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\include"
xcopy /Y "%SRC_DIR%\temp_build\NvToolsExt\lib\x64\*.*" "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\lib\x64"

echo Installing cuDNN...
7z x %CUDNN_SETUP_FILE% -o"%SRC_DIR%\temp_build\cudnn"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\cuda\bin\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\cuda\lib\x64\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\lib\x64"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\cuda\include\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\include"

echo Setting up environment...
set "PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\libnvvp;%PATH%"
set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%"
set "CUDA_PATH_V%CUDA_VER_MAJOR%_%CUDA_VER_MINOR%=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%"
set "NVTOOLSEXT_PATH=%ProgramFiles%\NVIDIA Corporation\NvToolsExt\"

echo Cleaning temp files
rd /s /q "%SRC_DIR%\temp_build"
pushd %TEMP%
rd /s /q .
popd

echo Using VS2015 as NVCC compiler
set "CUDAHOSTCXX=%VS140COMNTOOLS%\..\..\VC\bin\amd64\cl.exe"
