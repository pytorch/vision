@echo off

set VS_INSTALLER=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe
set VS_INSTALL_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise
set VS_INSTALL_ARGS=--nocache --quiet --wait --add Microsoft.VisualStudio.Component.VC.Tools.14.11
set VSDEVCMD_ARGS=-vcvars_ver=14.11

curl -k -L %VS_DOWNLOAD_LINK% --output vs_installer.exe
if errorlevel 1 exit /b 1

start /wait "%VS_INSTALLER%" modify --installPath "%VS_INSTALL_PATH%" %VS_INSTALL_ARGS%
if not errorlevel 0 exit /b 1
if errorlevel 1 if not errorlevel 3010 exit /b 1
if errorlevel 3011 exit /b 1
