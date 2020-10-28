pushd ext_deps
pushd machomachomangler-master
"%PYTHON%" -m pip install .
if errorlevel 1 exit 1
popd
popd

"%PYTHON%" setup.py install --single-version-externally-managed --record=record.txt

"%PYTHON%" -m pip uninstall machomachomangler
