#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env
pytest --cov=torchvision --junitxml=test-results/junit.xml -v --durations 20 test --ignore=test/test_datasets_download.py