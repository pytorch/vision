set -ex

if [[ $(uname) == "Linux" ]]; then
    pushd ext_deps
    pushd pyelftools-0.26
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
    popd
fi
