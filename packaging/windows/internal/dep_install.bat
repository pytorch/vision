@echo off

REM curl -k https://www.7-zip.org/a/7z1805-x64.exe -O
REM if errorlevel 1 exit /b 1

REM start /wait 7z1805-x64.exe /S
REM if errorlevel 1 exit /b 1

REM set "PATH=%ProgramFiles%\7-Zip;%PATH%"

choco feature disable --name showDownloadProgress
choco feature enable --name allowGlobalConfirmation

choco install curl 7zip
