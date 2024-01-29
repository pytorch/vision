#!/usr/bin/env bash

set -euo pipefail

./.github/scripts/setup-env.sh

# Activate conda environment
eval "$($(which conda) shell.bash hook)" && conda deactivate && conda activate ci

echo '::group::Install testing utilities'
pip install --progress-bar=off "pytest<8" pytest-mock pytest-cov expecttest
echo '::endgroup::'

python test/smoke_test.py
pytest --junit-xml="${RUNNER_TEST_RESULTS_DIR}/test-results.xml" -v --durations=25
