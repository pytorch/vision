#!/bin/bash
LD_LIBRARY_PATH="/usr/local/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH" python packaging/wheel/relocate.py

if [[ "$(uname)" == "Linux" ]]; then
    extra_decoders_channel="--pre --index-url https://download.pytorch.org/whl/nightly/cpu"
else
    extra_decoders_channel=""
fi

pip install torchvision-extra-decoders $extra_decoders_channel
