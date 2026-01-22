#!/bin/bash
set -euxo pipefail

if [ -n "${CUDA_HOME:-}" ]; then
    LD_LIBRARY_PATH="/usr/local/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
fi

python packaging/wheel/relocate.py

if [[ "$(uname)" == "Linux" && "$(uname -m)" != "aarch64" ]]; then
    extra_decoders_channel="--pre --index-url https://download.pytorch.org/whl/nightly/cpu"
else
    extra_decoders_channel=""
fi

pip install torchvision-extra-decoders $extra_decoders_channel
