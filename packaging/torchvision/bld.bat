pushd ext_deps
pushd machomachomangler-master
"%PYTHON%" setup.py install
if errorlevel 1 exit 1
popd
popd

"%PYTHON%" setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1

"%PYTHON%" -m pip uninstall machomachomangler -y
if errorlevel 1 exit 1
