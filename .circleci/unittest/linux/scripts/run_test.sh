#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

PYTORCH_JIT_ENABLE_NVFUSER=1 python -m torch.utils.collect_env
pytest --junitxml=test-results/junit.xml -v --durations 20
