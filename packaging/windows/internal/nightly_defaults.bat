@echo on

if "%~1"=="" goto arg_error
if NOT "%~2"=="" goto arg_error
goto arg_end

:arg_error

echo Illegal number of parameters. Pass packge type `Conda` or `Wheels`.
exit /b 1

:arg_end

echo "nightly_defaults.bat at %CD% starting at %DATE%"

set SRC_DIR=%~dp0\..

:: NIGHTLIES_FOLDER
:: N.B. this is also defined in cron_start.sh
::   An arbitrary root folder to store all nightlies folders, each of which is a
::   parent level date folder with separate subdirs for logs, wheels, conda
::   packages, etc. This should be kept the same across all scripts called in a
::   cron job, so it only has a default value in the top-most script
::   build_cron.sh to avoid the default values from diverging.
if "%NIGHTLIES_FOLDER%" == "" set "NIGHTLIES_FOLDER=%SRC_DIR%"

:: NIGHTLIES_DATE
:: N.B. this is also defined in cron_start.sh
::   The date in YYYY_mm_dd format that we are building for. If this is not
::   already set, then this will first try to find the date of the nightlies
::   folder that this builder repo exists in; e.g. if this script exists in
::   some_dir/2019_09_04/builder/cron/ then this will be set to 2019_09_04 (must
::   match YYYY_mm_dd). This is for convenience when debugging/uploading past
::   dates, so that you don't have to set NIGHTLIES_DATE yourself. If a date
::   folder cannot be found in that exact location, then this will default to
::   the current date.


if "%NIGHTLIES_DATE%" == "" ( goto date_start ) else ( goto date_end )

:date_start

set "DATE_CMD=Get-Date ([System.TimeZoneInfo]::ConvertTimeFromUtc((Get-Date).ToUniversalTime(), [System.TimeZoneInfo]::FindSystemTimeZoneById('Pacific Standard Time'))) -f 'yyyy_MM_dd'"
set "DATE_COMPACT_CMD=Get-Date ([System.TimeZoneInfo]::ConvertTimeFromUtc((Get-Date).ToUniversalTime(), [System.TimeZoneInfo]::FindSystemTimeZoneById('Pacific Standard Time'))) -f 'yyyyMMdd'"

FOR /F "delims=" %%i IN ('powershell -c "%DATE_CMD%"') DO set NIGHTLIES_DATE=%%i
FOR /F "delims=" %%i IN ('powershell -c "%DATE_COMPACT_CMD%"') DO set NIGHTLIES_DATE_COMPACT=%%i

:date_end

if "%NIGHTLIES_DATE_COMPACT%" == "" set NIGHTLIES_DATE_COMPACT=%NIGHTLIES_DATE:~0,4%%NIGHTLIES_DATE:~5,2%%NIGHTLIES_DATE:~8,2%

:: Used in lots of places as the root dir to store all conda/wheel/manywheel
:: packages as well as logs for the day
set today=%NIGHTLIES_FOLDER%\%NIGHTLIES_DATE%
mkdir "%today%" || ver >nul


::#############################################################################
:: Add new configuration variables below this line. 'today' should always be
:: defined ASAP to avoid weird errors
::#############################################################################


:: List of people to email when things go wrong. This is passed directly to
:: `mail -t`
:: TODO: Not supported yet
if "%NIGHTLIES_EMAIL_LIST%" == "" set NIGHTLIES_EMAIL_LIST=peterghost86@gmail.com

:: PYTORCH_CREDENTIALS_FILE
::   A bash file that exports credentials needed to upload to aws and anaconda.
::   Needed variables are PYTORCH_ANACONDA_USERNAME, PYTORCH_ANACONDA_PASSWORD,
::   AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY. Or it can just export the AWS
::   keys and then prepend a logged-in conda installation to the path.
:: TODO: Not supported yet
if "%PYTORCH_CREDENTIALS_FILE%" == "" set PYTORCH_CREDENTIALS_FILE=/c/Users/administrator/nightlies/credentials.sh

:: Location of the temporary miniconda that is downloaded to install conda-build
:: and aws to upload finished packages TODO this is messy to install this in
:: upload.sh and later use it in upload_logs.sh
if "%CONDA_UPLOADER_INSTALLATION%" == "" set "CONDA_UPLOADER_INSTALLATION=%today%\miniconda"

:: N.B. BUILDER_REPO and BUILDER_BRANCH are both set in cron_start.sh, as that
:: is the script that actually clones the builder repo that /this/ script is
:: running from.
pushd "%SRC_DIR%\.."
set NIGHTLIES_BUILDER_ROOT=%CD%
popd

:: The shared pytorch repo to be used by all builds
if "%NIGHTLIES_PYTORCH_ROOT%" == "" set "NIGHTLIES_PYTORCH_ROOT=%today%\vision"

:: PYTORCH_REPO
::   The Github org/user whose fork of Pytorch to check out (git clone
::   https://github.com/<THIS_PART>/pytorch.git). This will always be cloned
::   fresh to build with. Default is 'pytorch'
if "%PYTORCH_REPO%" == "" set PYTORCH_REPO=pytorch

:: PYTORCH_BRANCH
::   The branch of Pytorch to checkout for building (git checkout <THIS_PART>).
::   This can either be the name of the branch (e.g. git checkout
::   my_branch_name) or can be a git commit (git checkout 4b2674n...). Default
::   is 'latest', which is a special term that signals to pull the last commit
::   before 0:00 midnight on the NIGHTLIES_DATE
if "%PYTORCH_BRANCH%" == "" set PYTORCH_BRANCH=nightly

:: Clone the requested pytorch checkout
if exist "%NIGHTLIES_PYTORCH_ROOT%" ( goto clone_end ) else ( goto clone_start )

:clone_start

git clone --recursive "https://github.com/%PYTORCH_REPO%/vision.git" "%NIGHTLIES_PYTORCH_ROOT%"
pushd "%NIGHTLIES_PYTORCH_ROOT%"

if "%PYTORCH_BRANCH%" == "latest" ( goto latest_start ) else ( goto latest_end )

:latest_start

:: Switch to the latest commit by 11:59 yesterday
echo PYTORCH_BRANCH is set to latest so I will find the last commit
echo before 0:00 midnight on %NIGHTLIES_DATE%
set git_date=%NIGHTLIES_DATE:_=-%
FOR /F "delims=" %%i IN ('git log --before %git_date% -n 1 "--pretty=%%H"') DO set last_commit=%%i
echo Setting PYTORCH_BRANCH to %last_commit% since that was the last
echo commit before %NIGHTLIES_DATE%
set PYTORCH_BRANCH=%last_commit%

:latest_end

git checkout "%PYTORCH_BRANCH%"
git submodule update
popd

:clone_end

if "%CUDA_VERSION%" == "cpu" (
    set _DESIRED_CUDA=cpu
) else (
    set _DESIRED_CUDA=cu%CUDA_VERSION%
)

:: PYTORCH_BUILD_VERSION
::   The actual version string. Used in conda like
::       pytorch-nightly==1.0.0.dev20180908
::   or in manylinux like
::       torch_nightly-1.0.0.dev20180908-cp27-cp27m-linux_x86_64.whl
if "%TORCHVISION_BUILD_VERSION%" == "" set TORCHVISION_BUILD_VERSION=0.7.0.dev%NIGHTLIES_DATE_COMPACT%

if "%~1" == "Wheels" (
    if not "%CUDA_VERSION%" == "102" (
        set TORCHVISION_BUILD_VERSION=%TORCHVISION_BUILD_VERSION%+%_DESIRED_CUDA%
    )
)

:: PYTORCH_BUILD_NUMBER
::   This is usually the number 1. If more than one build is uploaded for the
::   same version/date, then this can be incremented to 2,3 etc in which case
::   '.post2' will be appended to the version string of the package. This can
::   be set to '0' only if OVERRIDE_PACKAGE_VERSION is being used to bypass
::   all the version string logic in downstream scripts. Since we use the
::   override below, exporting this shouldn't actually matter.
if "%TORCHVISION_BUILD_NUMBER%" == "" set /a TORCHVISION_BUILD_NUMBER=1
if %TORCHVISION_BUILD_NUMBER% GTR 1 set TORCHVISION_BUILD_VERSION=%TORCHVISION_BUILD_VERSION%%TORCHVISION_BUILD_NUMBER%

:: The nightly builds use their own versioning logic, so we override whatever
:: logic is in setup.py or other scripts
:: TODO: Not supported yet
set OVERRIDE_PACKAGE_VERSION=%TORCHVISION_BUILD_VERSION%
set BUILD_VERSION=%TORCHVISION_BUILD_VERSION%

:: Build folder for conda builds to use
if "%TORCH_CONDA_BUILD_FOLDER%" == "" set TORCH_CONDA_BUILD_FOLDER=torchvision

:: TORCH_PACKAGE_NAME
::   The name of the package to upload. This should probably be pytorch or
::   pytorch-nightly. N.B. that pip will change all '-' to '_' but conda will
::   not. This is dealt with in downstream scripts.
:: TODO: Not supported yet
if "%TORCH_PACKAGE_NAME%" == "" set TORCH_PACKAGE_NAME=torchvision

:: PIP_UPLOAD_FOLDER should end in a slash. This is to handle it being empty
:: (when uploading to e.g. whl/cpu/) and also to handle nightlies (when
:: uploading to e.g. /whl/nightly/cpu)
:: TODO: Not supported yet
if "%PIP_UPLOAD_FOLDER%" == "" set "PIP_UPLOAD_FOLDER=nightly\"

:: The location of the binary_sizes dir in s3 is hardcoded into
:: upload_binary_sizes.sh

:: DAYS_TO_KEEP
::   How many days to keep around for clean.sh. Build folders older than this
::   will be purged at the end of cron jobs. '1' means to keep only the current
::   day. Values less than 1 are not allowed. The default is 5.
:: TODO: Not supported yet
if "%DAYS_TO_KEEP%" == "" set /a DAYS_TO_KEEP=5
if %DAYS_TO_KEEP% LSS 1 (
    echo DAYS_TO_KEEP cannot be less than 1.
    echo A value of 1 means to only keep the build for today
    exit /b 1
)
