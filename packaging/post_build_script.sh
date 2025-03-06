#!/bin/bash
pip install auditwheel

if [[ ${ARCH} == "aarch64" ]]; then
    auditwheel repair --plat manylinux_2_28_aarch64 dist/*.whl
fi

LD_LIBRARY_PATH="/usr/local/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH" python packaging/wheel/relocate.py

pip install torchvision-extra-decoders
