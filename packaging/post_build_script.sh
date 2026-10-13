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

# torchvision-extra-decoders is not yet published for Python 3.15 / 3.15t
# (both the standard and free-threaded builds report version 3.15), so skip
# installing it there until wheels are available.
if python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 15) else 1)"; then
    echo "Skipping torchvision-extra-decoders: no Python 3.15 wheel published yet"
else
    pip install torchvision-extra-decoders $extra_decoders_channel
fi
