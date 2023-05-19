@echo on

if "%CU_VERSION%" == "cpu" (
    echo Skipping for CPU builds
    exit /b 0
)

set SRC_DIR=%~dp0\..

if not exist "%SRC_DIR%\temp_build" mkdir "%SRC_DIR%\temp_build"

rem in unit test workflow, we get CUDA_VERSION, for example 11.1
if defined CUDA_VERSION (
    set CUDA_VER=%CUDA_VERSION:.=%
) else (
    set CUDA_VER=%CU_VERSION:cu=%
)

set /a CUDA_VER=%CU_VERSION:cu=%
set CUDA_VER_MAJOR=%CUDA_VER:~0,-1%
set CUDA_VER_MINOR=%CUDA_VER:~-1,1%
set CUDA_VERSION_STR=%CUDA_VER_MAJOR%.%CUDA_VER_MINOR%
set CUDNN_FOLDER="cuda"
set CUDNN_LIB_FOLDER="lib\x64"

if %CUDA_VER% EQU 117 goto cuda117
if %CUDA_VER% EQU 118 goto cuda118
if %CUDA_VER% EQU 121 goto cuda121

echo CUDA %CUDA_VERSION_STR% is not supported
exit /b 1

:cuda117

set CUDA_INSTALL_EXE=cuda_11.7.0_516.01_windows.exe
if not exist "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%" (
    curl -k -L "https://ossci-windows.s3.amazonaws.com/%CUDA_INSTALL_EXE%" --output "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    set "ARGS=thrust_11.7 nvcc_11.7 cuobjdump_11.7 nvprune_11.7 nvprof_11.7 cupti_11.7 cublas_11.7 cublas_dev_11.7 cudart_11.7 cufft_11.7 cufft_dev_11.7 curand_11.7 curand_dev_11.7 cusolver_11.7 cusolver_dev_11.7 cusparse_11.7 cusparse_dev_11.7 npp_11.7 npp_dev_11.7 nvjpeg_11.7 nvjpeg_dev_11.7 nvrtc_11.7 nvrtc_dev_11.7 nvml_dev_11.7"
)

set CUDNN_INSTALL_ZIP=cudnn-windows-x86_64-8.3.2.44_cuda11.5-archive.zip
set CUDNN_FOLDER=cudnn-windows-x86_64-8.3.2.44_cuda11.5-archive
set CUDNN_LIB_FOLDER="lib"
if not exist "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%" (
    curl -k -L "http://s3.amazonaws.com/ossci-windows/%CUDNN_INSTALL_ZIP%" --output "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"

    rem Make sure windows path contains zlib dll
    curl -k -L "http://s3.amazonaws.com/ossci-windows/zlib123dllx64.zip" --output "%SRC_DIR%\temp_build\zlib123dllx64.zip"
    7z x "%SRC_DIR%\temp_build\zlib123dllx64.zip" -o"%SRC_DIR%\temp_build\zlib"
    xcopy /Y "%SRC_DIR%\temp_build\zlib\dll_x64\*.dll" "C:\Windows\System32"
)

goto cuda_common

:cuda118

set CUDA_INSTALL_EXE=cuda_11.8.0_522.06_windows.exe
if not exist "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%" (
    curl -k -L "https://ossci-windows.s3.amazonaws.com/%CUDA_INSTALL_EXE%" --output "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    set "ARGS=cuda_profiler_api_11.8 thrust_11.8 nvcc_11.8 cuobjdump_11.8 nvprune_11.8 nvprof_11.8 cupti_11.8 cublas_11.8 cublas_dev_11.8 cudart_11.8 cufft_11.8 cufft_dev_11.8 curand_11.8 curand_dev_11.8 cusolver_11.8 cusolver_dev_11.8 cusparse_11.8 cusparse_dev_11.8 npp_11.8 npp_dev_11.8 nvjpeg_11.8 nvjpeg_dev_11.8 nvrtc_11.8 nvrtc_dev_11.8 nvml_dev_11.8"
)

set CUDNN_INSTALL_ZIP=cudnn-windows-x86_64-8.5.0.96_cuda11-archive.zip
set CUDNN_FOLDER=cudnn-windows-x86_64-8.5.0.96_cuda11-archive
set CUDNN_LIB_FOLDER="lib"
if not exist "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%" (
    curl -k -L "http://s3.amazonaws.com/ossci-windows/%CUDNN_INSTALL_ZIP%" --output "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"

    rem Make sure windows path contains zlib dll
    curl -k -L "http://s3.amazonaws.com/ossci-windows/zlib123dllx64.zip" --output "%SRC_DIR%\temp_build\zlib123dllx64.zip"
    7z x "%SRC_DIR%\temp_build\zlib123dllx64.zip" -o"%SRC_DIR%\temp_build\zlib"
    xcopy /Y "%SRC_DIR%\temp_build\zlib\dll_x64\*.dll" "C:\Windows\System32"
)

goto cuda_common

:cuda121

set CUDA_INSTALL_EXE=cuda_12.1.0_531.14_windows.exe
if not exist "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%" (
    curl -k -L "https://ossci-windows.s3.amazonaws.com/%CUDA_INSTALL_EXE%" --output "%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    if errorlevel 1 exit /b 1
    set "CUDA_SETUP_FILE=%SRC_DIR%\temp_build\%CUDA_INSTALL_EXE%"
    set "ARGS=cuda_profiler_api_12.1 thrust_12.1 nvcc_12.1 cuobjdump_12.1 nvprune_12.1 nvprof_12.1 cupti_12.1 cublas_12.1 cublas_dev_12.1 cudart_12.1 cufft_12.1 cufft_dev_12.1 curand_12.1 curand_dev_12.1 cusolver_12.1 cusolver_dev_12.1 cusparse_12.1 cusparse_dev_12.1 npp_12.1 npp_dev_12.1 nvrtc_12.1 nvrtc_dev_12.1 nvml_dev_12.1"
)

set CUDNN_FOLDER=cudnn-windows-x86_64-8.8.1.3_cuda12-archive
set "CUDNN_INSTALL_ZIP=%CUDNN_FOLDER%.zip"
set CUDNN_LIB_FOLDER="lib"
if not exist "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%" (
    curl -k -L "http://s3.amazonaws.com/ossci-windows/%CUDNN_INSTALL_ZIP%" --output "%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"
    if errorlevel 1 exit /b 1
    set "CUDNN_SETUP_FILE=%SRC_DIR%\temp_build\%CUDNN_INSTALL_ZIP%"

    rem Make sure windows path contains zlib dll
    curl -k -L "http://s3.amazonaws.com/ossci-windows/zlib123dllx64.zip" --output "%SRC_DIR%\temp_build\zlib123dllx64.zip"
    7z x "%SRC_DIR%\temp_build\zlib123dllx64.zip" -o"%SRC_DIR%\temp_build\zlib"
    xcopy /Y "%SRC_DIR%\temp_build\zlib\dll_x64\*.dll" "C:\Windows\System32"
)

goto cuda_common

:cuda_common

if not exist "%SRC_DIR%\temp_build\NvToolsExt.7z" (
    curl -k -L https://www.dropbox.com/s/9mcolalfdj4n979/NvToolsExt.7z?dl=1 --output "%SRC_DIR%\temp_build\NvToolsExt.7z"
    if errorlevel 1 exit /b 1
)

echo Installing CUDA toolkit...
7z x %CUDA_SETUP_FILE% -o"%SRC_DIR%\temp_build\cuda"
pushd "%SRC_DIR%\temp_build\cuda"
sc config wuauserv start= disabled
sc stop wuauserv
sc query wuauserv

start /wait setup.exe -s %ARGS% -loglevel:6 -log:"%cd%/cuda_install_logs"
echo %errorlevel%

popd

echo Installing VS integration...
rem It's for VS 2019
xcopy /YI "%SRC_DIR%\temp_build\cuda\visual_studio_integration\CUDAVisualStudioIntegration\extras\visual_studio_integration\MSBuildExtensions\*.*" "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations"


echo Installing NvToolsExt...
7z x %SRC_DIR%\temp_build\NvToolsExt.7z -o"%SRC_DIR%\temp_build\NvToolsExt"
mkdir "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\bin\x64"
mkdir "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\include"
mkdir "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\lib\x64"
xcopy /Y "%SRC_DIR%\temp_build\NvToolsExt\bin\x64\*.*" "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\bin\x64"
xcopy /Y "%SRC_DIR%\temp_build\NvToolsExt\include\*.*" "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\include"
xcopy /Y "%SRC_DIR%\temp_build\NvToolsExt\lib\x64\*.*" "%ProgramFiles%\NVIDIA Corporation\NvToolsExt\lib\x64"

echo Setting up environment...
set "PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\libnvvp;%PATH%"
set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%"
set "CUDA_PATH_V%CUDA_VER_MAJOR%_%CUDA_VER_MINOR%=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%"
set "NVTOOLSEXT_PATH=%ProgramFiles%\NVIDIA Corporation\NvToolsExt\bin\x64"

if not exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin\nvcc.exe" (
    echo CUDA %CUDA_VERSION_STR% installed failed.
    echo --------- RunDll32.exe.log
    type "%SRC_DIR%\temp_build\cuda\cuda_install_logs\LOG.RunDll32.exe.log"
    echo --------- setup.exe.log -------
    type "%SRC_DIR%\temp_build\cuda\cuda_install_logs\LOG.setup.exe.log"
    exit /b 1
)

echo Installing cuDNN...
7z x %CUDNN_SETUP_FILE% -o"%SRC_DIR%\temp_build\cudnn"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\%CUDNN_FOLDER%\bin\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\bin"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\%CUDNN_FOLDER%\%CUDNN_LIB_FOLDER%\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\lib\x64"
xcopy /Y "%SRC_DIR%\temp_build\cudnn\%CUDNN_FOLDER%\include\*.*" "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION_STR%\include"

echo Cleaning temp files
rd /s /q "%SRC_DIR%\temp_build" || ver > nul
