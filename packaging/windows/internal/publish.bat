@echo off

set SRC_DIR=%~dp0
pushd %SRC_DIR%

if NOT "%CUDA_VERSION%" == "cpu" (
    set PACKAGE_SUFFIX=_cuda%CUDA_VERSION%
) else (
    set PACKAGE_SUFFIX=
)

if "%PACKAGEFULLNAME%" == "Conda" (
    set PACKAGE=conda
) else (
    set PACKAGE=wheels
)

set PUBLISH_BRANCH=%PACKAGE%_%DESIRED_PYTHON%%PACKAGE_SUFFIX%

git clone %ARTIFACT_REPO_URL% -b %PUBLISH_BRANCH% --single-branch >nul 2>&1

IF ERRORLEVEL 1 (
    echo Branch %PUBLISH_BRANCH% not exist, falling back to master
    set NO_BRANCH=1
    git clone %ARTIFACT_REPO_URL% -b master --single-branch >nul 2>&1
)

IF ERRORLEVEL 1 (
    echo Clone failed
    goto err
)

cd pytorch_builder
attrib -s -h -r . /s /d

:: Empty repo
rd /s /q . || ver >nul

IF NOT EXIST %PACKAGE% mkdir %PACKAGE%

xcopy /S /E /Y ..\..\output\*.* %PACKAGE%\

git config --global user.name "Azure DevOps"
git config --global user.email peterghost86@gmail.com
git init
git checkout --orphan %PUBLISH_BRANCH%
git remote add origin %ARTIFACT_REPO_URL%
git add .
git commit -m "Update artifacts"

:push

if "%RETRY_TIMES%" == "" (
    set /a RETRY_TIMES=10
    set /a SLEEP_TIME=2
) else (
    set /a RETRY_TIMES=%RETRY_TIMES%-1
    set /a SLEEP_TIME=%SLEEP_TIME%*2
)

git push origin %PUBLISH_BRANCH% -f > nul 2>&1

IF ERRORLEVEL 1 (
    echo Git push retry times remaining: %RETRY_TIMES%
    echo Sleep time: %SLEEP_TIME% seconds
    IF %RETRY_TIMES% EQU 0 (
        echo Push failed
        goto err
    )
    waitfor SomethingThatIsNeverHappening /t %SLEEP_TIME% 2>nul || ver >nul
    goto push
) ELSE (
    set RETRY_TIMES=
    set SLEEP_TIME=
)

popd

exit /b 0

:err

popd

exit /b 1
