@echo off

:: Caution: Please don't use this script locally
:: It may destroy your build environment.

setlocal

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

call "%VS15VCVARSALL%" x86_amd64
for /f "usebackq tokens=*" %%i in (`where link.exe`) do move "%%i" "%%i.bak"

endlocal
