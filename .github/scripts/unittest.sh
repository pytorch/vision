#!/usr/bin/env bash

set -euo pipefail

./.github/scripts/setup-env.sh

# Activate conda environment
eval "$($(which conda) shell.bash hook)" && conda deactivate && conda activate ci

# Fix SSL certificate verification on Windows by using certifi's CA bundle
if [[ "$(uname)" == MSYS* ]]; then
    export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
fi

echo '::group::Install testing utilities'
# TODO: remove the <8 constraint on pytest when https://github.com/pytorch/vision/issues/8238 is closed
pip install --progress-bar=off "pytest<8" pytest-mock pytest-cov expecttest!=0.2.0 requests
echo '::endgroup::'

python test/smoke_test.py

# We explicitly ignore the video tests until we resolve https://github.com/pytorch/vision/issues/8162
pytest --ignore-glob="*test_video*" --ignore-glob="*test_onnx*" --junit-xml="${RUNNER_TEST_RESULTS_DIR}/test-results.xml" -v --durations=25 -k "not TestFxFeatureExtraction"
