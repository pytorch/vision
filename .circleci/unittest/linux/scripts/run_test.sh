#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env

case "$(uname -s)" in
  Darwin*)
    # The largest macOS runner is not able to handle the regular test suite plus the transforms v2 tests at the same
    # time due to insufficient resources. Thus, we ignore the transforms v2 tests at first and run them in a separate
    # step afterwards.
    GLOB='test/test_transforms_v2*'
    pytest --junitxml=test-results/junit.xml -v --durations 20 --ignore-glob="${GLOB}"
    eval "pytest --junitxml=test-results/junit-transforms-v2.xml -v --durations 20 ${GLOB}"
    ;;
  *)
    pytest --junitxml=test-results/junit.xml -v --durations 20
    ;;
esac
