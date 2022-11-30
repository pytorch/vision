#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env
pytest --ignore=test/test_models.py --ignore=test/test_backbone_utils.py --junitxml=test-results/junit.xml -v --durations 5000
