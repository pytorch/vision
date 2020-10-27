set -ex

if [[ $(uname) == "Linux" ]]; then
    # Install PyELFtools manually since it is not available on conda
    pushd ext_deps
    pushd pyelftools-0.26
    python setup.py install --single-version-externally-managed --record files.txt
    popd
    popd
    export LD_LIBRARY_PATH=$BUILD_PREFIX/lib:$LD_LIBRARY_PATH
fi

python setup.py install --single-version-externally-managed --record=record.txt

if [[ $(uname) == "Linux" ]]; then
    # Uninstall PyELFtools
    pushd ext_deps
    pushd pyelftools-0.26
    cat files.txt | xargs rm -rf
    popd
    popd
fi
