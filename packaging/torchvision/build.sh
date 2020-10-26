set -ex

if [[ $(uname) == "Linux" ]]; then
    pushd ext_deps
    pushd pyelftools-0.26
    python setup.py install --single-version-externally-managed --record files.txt
    popd
    pushd auditwheel-3.2.0
    python setup.py install --single-version-externally-managed --record files.txt
    popd
    popd
fi

python setup.py install --single-version-externally-managed --record=record.txt

if [[ $(uname) == "Linux" ]]; then
    pushd ext_deps
    pushd pyelftools-0.26
    cat files.txt | xargs rm -rf
    popd
    pushd auditwheel-3.2.0
    cat files.txt | xargs rm -rf
    popd
    popd
fi
