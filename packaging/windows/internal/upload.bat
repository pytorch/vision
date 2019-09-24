@echo off

IF "%CONDA_UPLOADER_INSTALLATION%" == "" goto precheck_fail
IF "%PYTORCH_FINAL_PACKAGE_DIR%" == "" goto precheck_fail
IF "%today%" == "" goto precheck_fail
IF "%PYTORCH_ANACONDA_USERNAME%" == "" goto precheck_fail
IF "%PYTORCH_ANACONDA_PASSWORD%" == "" goto precheck_fail

goto precheck_pass

:precheck_fail

echo Please run nightly_defaults.bat first.
echo And remember to set `PYTORCH_FINAL_PACKAGE_DIR`
echo Finally, don't forget to set anaconda tokens
exit /b 1

:precheck_pass

pushd %today%

:: Install anaconda client
set "CONDA_HOME=%CONDA_UPLOADER_INSTALLATION%"
set "tmp_conda=%CONDA_HOME%"
set "miniconda_exe=%CD%\miniconda.exe"
rmdir /s /q "%CONDA_HOME%"
del miniconda.exe
curl -k https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe -o "%miniconda_exe%"
popd

IF ERRORLEVEL 1 (
    echo Conda download failed
    exit /b 1
)

call %~dp0\..\..\conda\install_conda.bat

IF ERRORLEVEL 1 (
    echo Conda installation failed
    exit /b 1
)

set "ORIG_PATH=%PATH%"
set "PATH=%CONDA_HOME%;%CONDA_HOME%\scripts;%CONDA_HOME%\Library\bin;%PATH%"

REM conda install -y anaconda-client
pip install git+https://github.com/peterjc123/anaconda-client.git@log_more_meaningfull_errors
IF ERRORLEVEL 1 (
    echo Anaconda client installation failed
    exit /b 1
)

set PYTORCH_FINAL_PACKAGE=
:: Upload all the packages under `PYTORCH_FINAL_PACKAGE_DIR`
FOR /F "delims=" %%i IN ('where /R %PYTORCH_FINAL_PACKAGE_DIR% *vision*.tar.bz2') DO (
    set "PYTORCH_FINAL_PACKAGE=%%i"
)

IF "%PYTORCH_FINAL_PACKAGE%" == "" (
    echo No package to upload
    exit /b 0
)

:upload

if "%RETRY_TIMES%" == "" (
    set /a RETRY_TIMES=10
    set /a SLEEP_TIME=2
) else (
    set /a RETRY_TIMES=%RETRY_TIMES%-1
    set /a SLEEP_TIME=%SLEEP_TIME%*2
)

REM bash -c "yes | anaconda login --username "%PYTORCH_ANACONDA_USERNAME%" --password "%PYTORCH_ANACONDA_PASSWORD%""
anaconda login --username "%PYTORCH_ANACONDA_USERNAME%" --password "%PYTORCH_ANACONDA_PASSWORD%"
IF ERRORLEVEL 1 (
    echo Anaconda client login failed
    exit /b 1
)

echo Uploading %PYTORCH_FINAL_PACKAGE% to Anaconda Cloud
anaconda upload "%PYTORCH_FINAL_PACKAGE%" -u pytorch-nightly --label main --force --no-progress

IF ERRORLEVEL 1 (
    echo Anaconda upload retry times remaining: %RETRY_TIMES%
    echo Sleep time: %SLEEP_TIME% seconds
    IF %RETRY_TIMES% EQU 0 (
        echo Upload failed
        exit /b 1
    )
    waitfor SomethingThatIsNeverHappening /t %SLEEP_TIME% 2>nul || ver >nul
    goto upload
) ELSE (
    set RETRY_TIMES=
    set SLEEP_TIME=
)
