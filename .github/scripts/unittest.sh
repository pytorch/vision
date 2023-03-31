#!/usr/bin/env bash

set -euo pipefail

./.github/scripts/setup-env.sh

# Prepare conda
CONDA_PATH=$(which conda)
eval "$(${CONDA_PATH} shell.bash hook)"
conda activate ci

python test/smoke_test.py

#echo '::group::Install testing utilities'
#pip install --progress-bar=off pytest pytest-mock pytest-cov
#echo '::endgroup::'
#
#pytest --junit-xml="${RUNNER_TEST_RESULTS_DIR}/test-results.xml" -v --durations=25
