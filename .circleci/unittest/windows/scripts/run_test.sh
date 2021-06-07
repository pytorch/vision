#!/usr/bin/env bash

set -e

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
pytest test/test_lol.py
