@echo off

echo The flags after configuring:
echo NO_CUDA=%NO_CUDA%
echo CMAKE_GENERATOR=%CMAKE_GENERATOR%
if "%NO_CUDA%"==""  echo CUDA_PATH=%CUDA_PATH%
if NOT "%CC%"==""   echo CC=%CC%
if NOT "%CXX%"==""  echo CXX=%CXX%
if NOT "%DISTUTILS_USE_SDK%"==""  echo DISTUTILS_USE_SDK=%DISTUTILS_USE_SDK%

set SRC_DIR=%~dp0\..

IF "%VSDEVCMD_ARGS%" == "" (
    call "%VS15VCVARSALL%" x64
) ELSE (
    call "%VS15VCVARSALL%" x64 %VSDEVCMD_ARGS%
)

pushd %SRC_DIR%

IF NOT exist "setup.py" (
    cd %MODULE_NAME%
)

if "%CXX%"=="sccache cl" (
    sccache --stop-server
    sccache --start-server
    sccache --zero-stats
)


if "%BUILD_PYTHONLESS%" == "" goto pytorch else goto libtorch

:libtorch
set VARIANT=shared-with-deps

mkdir libtorch
mkdir libtorch\bin
mkdir libtorch\cmake
mkdir libtorch\include
mkdir libtorch\lib
mkdir libtorch\share
mkdir libtorch\test

mkdir build
pushd build
python ../tools/build_libtorch.py
popd

IF ERRORLEVEL 1 exit /b 1
IF NOT ERRORLEVEL 0 exit /b 1

move /Y torch\bin\*.* libtorch\bin\
move /Y torch\cmake\*.* libtorch\cmake\
robocopy /move /e torch\include\ libtorch\include\
move /Y torch\lib\*.* libtorch\lib\
robocopy /move /e torch\share\ libtorch\share\
move /Y torch\test\*.* libtorch\test\

move /Y libtorch\bin\*.dll libtorch\lib\

git rev-parse HEAD > libtorch\build-hash

IF "%DEBUG%" == "" (
    set LIBTORCH_PREFIX=libtorch-win-%VARIANT%
) ELSE (
    set LIBTORCH_PREFIX=libtorch-win-%VARIANT%-debug
)

7z a -tzip %LIBTORCH_PREFIX%-%PYTORCH_BUILD_VERSION%.zip libtorch\*

mkdir ..\output\%CUDA_PREFIX%
copy /Y %LIBTORCH_PREFIX%-%PYTORCH_BUILD_VERSION%.zip ..\output\%CUDA_PREFIX%\
copy /Y %LIBTORCH_PREFIX%-%PYTORCH_BUILD_VERSION%.zip ..\output\%CUDA_PREFIX%\%LIBTORCH_PREFIX%-latest.zip

goto build_end

:pytorch
:: This stores in e.g. D:/_work/1/s/windows/output/cpu
pip wheel -e . --no-deps --wheel-dir ../output/%CUDA_PREFIX%

:build_end
IF ERRORLEVEL 1 exit /b 1
IF NOT ERRORLEVEL 0 exit /b 1

if "%CXX%"=="sccache cl" (
    taskkill /im sccache.exe /f /t || ver > nul
    taskkill /im nvcc.exe /f /t || ver > nul
)

cd ..
