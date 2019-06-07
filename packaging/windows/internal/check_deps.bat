@echo off

REM Check for necessary components

IF NOT "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
    echo You should use 64 bits Windows to build and run PyTorch
    exit /b 1
)

IF "%BUILD_VISION%" == "" (
    where /q cmake.exe

    IF ERRORLEVEL 1 (
        echo CMake is required to compile PyTorch on Windows
        exit /b 1
    )
)

IF NOT EXIST "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    echo Visual Studio 2017 C++ BuildTools is required to compile PyTorch on Windows
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -legacy -products * -version [15^,16^) -property installationPath`) do (
    if exist "%%i" if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS15INSTALLDIR=%%i"
        set "VS15VCVARSALL=%%i\VC\Auxiliary\Build\vcvarsall.bat"
        goto vswhere
    )
)

:vswhere
IF "%VS15VCVARSALL%"=="" (
    echo Visual Studio 2017 C++ BuildTools is required to compile PyTorch on Windows
    exit /b 1
)

set MSSdk=1
set DISTUTILS_USE_SDK=1

where /q python.exe

IF ERRORLEVEL 1 (
    echo Python x64 3.5 or up is required to compile PyTorch on Windows
    exit /b 1
)

for /F "usebackq delims=" %%i in (`python -c "import sys; print('{0[0]}{0[1]}'.format(sys.version_info))"`) do (
    set /a PYVER=%%i
)

if  %PYVER% LSS 35 (
    echo Warning: PyTorch for Python 2 under Windows is experimental.
    echo Python x64 3.5 or up is recommended to compile PyTorch on Windows
    echo Maybe you can create a virual environment if you have conda installed:
    echo ^> conda create -n test python=3.6 pyyaml mkl numpy
    echo ^> activate test
)

for /F "usebackq delims=" %%i in (`python -c "import struct;print( 8 * struct.calcsize('P'))"`) do (
    set /a PYSIZE=%%i
)

if %PYSIZE% NEQ 64 (
    echo Python x64 3.5 or up is required to compile PyTorch on Windows
    exit /b 1
)
