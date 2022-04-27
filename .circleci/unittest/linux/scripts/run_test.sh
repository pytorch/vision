#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env
PYTORCH_NVFUSER_DISABLE_FALLBACK=1 PYTORCH_JIT_LOG_LEVEL=">>graph_fuser" pytest --junitxml=test-results/junit.xml -v --durations 20
