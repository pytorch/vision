@echo off

: From the following doc, the build won't be triggered if the users don't sign in daily.
: https://docs.microsoft.com/en-us/azure/devops/pipelines/build/triggers?tabs=yaml&view=vsts#my-build-didnt-run-what-happened
: To avoid this problem, we can just go through the sign in process using the following command.

:auth_start

if "%RETRY_TIMES%" == "" (
    set /a RETRY_TIMES=10
    set /a SLEEP_TIME=2
) else (
    set /a RETRY_TIMES=%RETRY_TIMES%-1
    set /a SLEEP_TIME=%SLEEP_TIME%*2
)

for /f "usebackq tokens=*" %%i in (`curl -so NUL -w "%%{http_code}" -u %VSTS_AUTH% https://dev.azure.com/pytorch`) do (
    set STATUS_CODE=%%i
)

IF NOT "%STATUS_CODE%" == "200" (
    echo Auth retry times remaining: %RETRY_TIMES%
    echo Sleep time: %SLEEP_TIME% seconds
    IF %RETRY_TIMES% EQU 0 (
        echo Auth failed
        goto err
    )
    waitfor SomethingThatIsNeverHappening /t %SLEEP_TIME% 2>nul || ver >nul
    goto auth_start
) ELSE (
    echo Login Attempt Succeeded
    goto auth_end
)

:err

: Throw a warning if it fails
powershell -c "Write-Warning 'Login Attempt Failed'"

:auth_end

set RETRY_TIMES=
set SLEEP_TIME=
set STATUS_CODE=

exit /b 0
