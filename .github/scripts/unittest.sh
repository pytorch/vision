#!/usr/bin/env bash

set -euo pipefail

./.github/scripts/setup-env.sh

# Prepare conda
eval "$($(which conda) shell.bash hook)"
conda deactivate && conda activate ci

python test/smoke_test.py

#echo '::group::Install testing utilities'
#pip install --progress-bar=off pytest pytest-mock pytest-cov
#echo '::endgroup::'
#
#pytest --junit-xml="${RUNNER_TEST_RESULTS_DIR}/test-results.xml" -v --durations=25
