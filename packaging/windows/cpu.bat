@echo off

IF NOT "%BUILD_VISION%" == "" (
    set MODULE_NAME=vision
) ELSE (
    set MODULE_NAME=pytorch
)

IF NOT EXIST "setup.py" IF NOT EXIST "%MODULE_NAME%" (
    call internal\clone.bat
    cd ..
    IF ERRORLEVEL 1 goto eof
) ELSE (
    call internal\clean.bat
)

call internal\check_deps.bat
IF ERRORLEVEL 1 goto eof

REM Check for optional components

echo Disabling CUDA
set NO_CUDA=1
set USE_CUDA=0

IF "%BUILD_VISION%" == "" (
    call internal\check_opts.bat
    IF ERRORLEVEL 1 goto eof

    call internal\copy_cpu.bat
    IF ERRORLEVEL 1 goto eof
)

call internal\setup.bat
IF ERRORLEVEL 1 goto eof

:eof
