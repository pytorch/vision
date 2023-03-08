#!/usr/bin/env bash

set -euo pipefail

./.github/scripts/setup-env.sh

# Prepare conda
CONDA_PATH=$(which conda)
eval "$(${CONDA_PATH} shell.bash hook)"
conda activate ci

echo '::group::Install testing utilities'
pip install --progress-bar=off pytest pytest-mock pytest-cov
echo '::endgroup::'

echo '::group::Run unittests'
pytest --durations=25
echo '::endgroup::'
