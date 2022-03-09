#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
pytest --junitxml=test-results/junit.xml -v --durations 2000 -k "test_classification_model or test_detection_model or test_quantized_classification_model or test_segmentation_model or test_video_model"
