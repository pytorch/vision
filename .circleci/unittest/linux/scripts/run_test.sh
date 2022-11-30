#!/usr/bin/env bash

set -e

PYTEST_ADDITIONAL_ARGS=$1

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env
pytest --junitxml=test-results/junit.xml -v --durations 20 ${PYTEST_ADDITIONAL_ARGS}
