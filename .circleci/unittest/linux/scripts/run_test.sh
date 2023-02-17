#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

python -m torch.utils.collect_env

case "$(uname -s)" in
    Darwin*) IGNORE='--ignore-glob=test/*v2*';;
    *) IGNORE=''
esac
pytest --junitxml=test-results/junit.xml -v --durations 20 $IGNORE
