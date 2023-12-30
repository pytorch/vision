#!/usr/bin/env bash

set -euo pipefail

./.github/scripts/setup-env.sh

# Activate conda environment
eval "$($(which conda) shell.bash hook)" && conda deactivate && conda activate ci

echo '::group::Install testing utilities'
pip install --progress-bar=off pytest pytest-mock pytest-cov expecttest!=0.2.0
echo '::endgroup::'

python test/smoke_test.py

# We explicitly ignore the video tests until we resolve https://github.com/pytorch/vision/issues/8162
pytest --ignore-glob="*test_video*" --junit-xml="${RUNNER_TEST_RESULTS_DIR}/test-results.xml" -v --durations=25
