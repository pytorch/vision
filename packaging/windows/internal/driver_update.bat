set "DRIVER_DOWNLOAD_LINK=https://ossci-windows.s3.amazonaws.com/461.09-data-center-tesla-desktop-winserver-2019-2016-international.exe"
curl --retry 3 -kL %DRIVER_DOWNLOAD_LINK% --output 461.09-data-center-tesla-desktop-winserver-2019-2016-international.exe
if errorlevel 1 exit /b 1

start /wait 461.09-data-center-tesla-desktop-winserver-2019-2016-international.exe -s -noreboot
if errorlevel 1 exit /b 1

del 461.09-data-center-tesla-desktop-winserver-2019-2016-international.exe || ver > NUL