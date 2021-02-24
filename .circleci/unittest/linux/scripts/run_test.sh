#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
if [ "${CU_VERSION:-}" == cpu ] ; then
    NUMPROCESSES="auto"
else
    NUMPROCESSES="1"
fi

python -m torch.utils.collect_env
pytest \
    --numprocesses=$NUMPROCESSES \
    --maxprocesses=2 \
    --cov=torchvision \
    --junitxml=test-results/junit.xml \
    --verbose \
    --durations 20 \
    --ignore=test/test_datasets_download.py \
    test
