@echo off

set VS_DOWNLOAD_LINK=https://aka.ms/vs/15/release/vs_enterprise.exe
set VS_INSTALL_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise
set VS_INSTALL_ARGS=--nocache --quiet --wait --add Microsoft.VisualStudio.Component.VC.Tools.14.11
set VSDEVCMD_ARGS=-vcvars_ver=14.11

curl -k -L %VS_DOWNLOAD_LINK% --output vs_installer.exe
if errorlevel 1 exit /b 1

start /wait vs_installer.exe modify --installPath "%VS_INSTALL_PATH%" %VS_INSTALL_ARGS%
if not errorlevel 0 exit /b 1
if errorlevel 1 if not errorlevel 3010 exit /b 1
if errorlevel 3011 exit /b 1
