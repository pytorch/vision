#!/usr/bin/env bash

set -euo pipefail

./.github/scripts/setup-env.sh

# Activate conda environment
eval "$($(which conda) shell.bash hook)" && conda deactivate && conda activate ci

echo '::group::Install testing utilities'
# TODO: remove the <8 constraint on pytest when https://github.com/pytorch/vision/issues/8238 is closed
pip install --progress-bar=off "pytest<8" pytest-mock pytest-cov expecttest!=0.2.0
echo '::endgroup::'

python test/smoke_test.py
pytest --color no --junit-xml="${RUNNER_TEST_RESULTS_DIR}/test-results.xml" test/test_transforms_v2.py -k kernel
