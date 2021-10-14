set YEAR=2019
set VER=16

mkdir "%PREFIX%\etc\conda\activate.d"
COPY "%RECIPE_DIR%\activate.bat" "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"

IF "%cross_compiler_target_platform%" == "win-64" (
  set "target_platform=amd64"
  echo SET "CMAKE_GENERATOR=Visual Studio %VER% %YEAR% Win64" >> "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"
  echo pushd "%%VSINSTALLDIR%%" >> "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"
  IF "%VSDEVCMD_ARGS%" == "" (
    echo CALL "VC\Auxiliary\Build\vcvarsall.bat" x64 >> "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"
    echo popd >> "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"
    echo pushd "%%VSINSTALLDIR%%" >> "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"
    echo CALL "VC\Auxiliary\Build\vcvarsall.bat" x86_amd64 >> "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"
  ) ELSE (
    echo CALL "VC\Auxiliary\Build\vcvarsall.bat" x64 %VSDEVCMD_ARGS% >> "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"
    echo popd >> "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"
    echo pushd "%%VSINSTALLDIR%%" >> "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"
    echo CALL "VC\Auxiliary\Build\vcvarsall.bat" x86_amd64 %VSDEVCMD_ARGS% >> "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"
  )
  echo popd >> "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"
  ) else (
  set "target_platform=x86"
  echo SET "CMAKE_GENERATOR=Visual Studio %VER% %YEAR%" >> "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"
  echo pushd "%%VSINSTALLDIR%%" >> "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"
  echo CALL "VC\Auxiliary\Build\vcvars32.bat" >> "%PREFIX%\etc\conda\activate.d\vs%YEAR%_compiler_vars.bat"
  echo popd
  )
