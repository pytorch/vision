#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

source .circleci/unittest/checktorch.sh

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
pytest --cov=torchvision --junitxml=test-results/junit.xml -v --durations 20
