if "%VC_YEAR%" == "2017" set VSDEVCMD_ARGS=-vcvars_ver=14.11
if "%VC_YEAR%" == "2017" powershell packaging/windows/internal/vs2017_install.ps1
if errorlevel 1 exit /b 1

call packaging/windows/internal/cuda_install.bat
if errorlevel 1 exit /b 1

call packaging/windows/internal/nightly_defaults.bat Wheels
if errorlevel 1 exit /b 1

call packaging/windows/build_vision.bat %CUDA_VERSION% %TORCHVISION_BUILD_VERSION% %TORCHVISION_BUILD_NUMBER%
if errorlevel 1 exit /b 1
