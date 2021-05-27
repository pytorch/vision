#!/usr/bin/env bash

set -e

eval "$(./conda/Scripts/conda.exe 'shell.bash' 'hook')"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
pytest --cov=torchvision --junitxml=test-results/junit.xml -v --durations 20 test --ignore=test/test_datasets_download.py -k "keypointrcnn_resnet50_fpn or maskrcnn_resnet50_fpn or fasterrcnn_resnet50_fpn or retinanet_resnet50_fpn"
