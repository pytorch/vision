set -ex

if [[ $(uname) == "Linux" ]]; then
    pip install auditwheel
fi

python setup.py install --single-version-externally-managed --record=record.txt

if [[ $(uname) == "Linux" ]]; then
    pip uninstall auditwheel
fi