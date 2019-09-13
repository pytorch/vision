@echo off

REM set VS_DOWNLOAD_LINK=https://aka.ms/vs/15/release/vs_buildtools.exe
REM IF "%VS_LATEST%" == "1" (
REM    set VS_INSTALL_ARGS= --nocache --norestart --quiet --wait --add Microsoft.VisualStudio.Workload.VCTools
REM    set VSDEVCMD_ARGS=
REM ) ELSE (
REM    set VS_INSTALL_ARGS=--nocache --quiet --wait --add Microsoft.VisualStudio.Workload.VCTools ^
REM                                                 --add Microsoft.VisualStudio.Component.VC.Tools.14.11 ^
REM                                                 --add Microsoft.Component.MSBuild ^
REM                                                 --add Microsoft.VisualStudio.Component.Roslyn.Compiler ^
REM                                                 --add Microsoft.VisualStudio.Component.TextTemplating ^
REM                                                 --add Microsoft.VisualStudio.Component.VC.CoreIde ^
REM                                                 --add Microsoft.VisualStudio.Component.VC.Redist.14.Latest ^
REM                                                 --add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Core ^
REM                                                 --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
REM                                                 --add Microsoft.VisualStudio.Component.VC.Tools.14.11 ^
REM                                                 --add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Win81
REM    set VSDEVCMD_ARGS=-vcvars_ver=14.11
REM )

set VS_DOWNLOAD_LINK=https://aka.ms/vs/15/release/ca16a813d/vs_buildtools.exe
set VS_INSTALL_ARGS=--nocache --quiet --wait --add Microsoft.VisualStudio.Workload.VCTools ^
                    --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
                    --add Microsoft.VisualStudio.Component.VC.DiagnosticTools ^
                    --add Microsoft.VisualStudio.Component.Windows10SDK.16299.Desktop ^
                    --add Microsoft.VisualStudio.Component.VC.CMake.Project ^
                    --add Microsoft.VisualStudio.Component.VC.ATL ^
                    --add Microsoft.VisualStudio.Component.VC.140

curl -k -L %VS_DOWNLOAD_LINK% --output vs_installer.exe
if errorlevel 1 exit /b 1

start /wait .\vs_installer.exe %VS_INSTALL_ARGS%
if not errorlevel 0 exit /b 1
if errorlevel 1 if not errorlevel 3010 exit /b 1
if errorlevel 3011 exit /b 1
