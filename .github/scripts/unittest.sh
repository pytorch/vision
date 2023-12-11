#!/usr/bin/env bash

set -euo pipefail

./.github/scripts/setup-env.sh

# Activate conda environment
eval "$($(which conda) shell.bash hook)" && conda deactivate && conda activate ci

echo '::group::Install testing utilities'
pip install --progress-bar=off pytest pytest-mock pytest-cov expecttest
echo '::endgroup::'

python test/smoke_test.py
pytest \
  --verbose --color no \
  --durations=25 \
  --junit-xml="${RUNNER_TEST_RESULTS_DIR}/test-results.xml" \
