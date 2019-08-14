set VC_PATH=x86
if "%ARCH%"=="64" (
   set VC_PATH=x64
)

set MSC_VER=2017

rem :: This should always be present for VC installed with VS.  Not sure about VC installed with Visual C++ Build Tools 2015
rem FOR /F "usebackq tokens=3*" %%A IN (`REG QUERY "HKEY_LOCAL_MACHINE\Software\Microsoft\DevDiv\VC\Servicing\14.0\IDE.x64" /v UpdateVersion`) DO (
rem     set SP=%%A
rem     )

rem if not "%SP%" == "%PKG_VERSION%" (
rem    echo "Version detected from registry: %SP%"
rem    echo    "does not match version of package being built (%PKG_VERSION%)"
rem    echo "Do you have current updates for VS 2015 installed?"
rem    exit 1
rem )


REM ========== REQUIRES Win 10 SDK be installed, or files otherwise copied to location below!
robocopy "C:\Program Files (x86)\Windows Kits\10\Redist\ucrt\DLLs\%VC_PATH%"  "%LIBRARY_BIN%" *.dll /E
robocopy "C:\Program Files (x86)\Windows Kits\10\Redist\ucrt\DLLs\%VC_PATH%"  "%PREFIX%" *.dll /E
if %ERRORLEVEL% GEQ 8 exit 1

REM ========== This one comes from visual studio 2017
set "VC_VER=141"

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -legacy -products * -version [15^,16^) -property installationPath`) do (
    if exist "%%i" if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS15VCVARSALL=%%i\VC\Auxiliary\Build\vcvarsall.bat"
        goto :eof
    )
)

@setlocal
call "%VS15VARSALL%" x64

set "REDIST_ROOT=%VCToolsRedistDir%%VC_PATH%"

robocopy "%REDIST_ROOT%\Microsoft.VC%VC_VER%.CRT" "%LIBRARY_BIN%" *.dll /E
if %ERRORLEVEL% LSS 8 exit 0
robocopy "%REDIST_ROOT%\Microsoft.VC%VC_VER%.CRT" "%PREFIX%" *.dll /E
if %ERRORLEVEL% LSS 8 exit 0
robocopy "%REDIST_ROOT%\Microsoft.VC%VC_VER%.OpenMP" "%LIBRARY_BIN%" *.dll /E
if %ERRORLEVEL% LSS 8 exit 0
robocopy "%REDIST_ROOT%\Microsoft.VC%VC_VER%.OpenMP" "%PREFIX%" *.dll /E
if %ERRORLEVEL% LSS 8 exit 0
@endlocal
