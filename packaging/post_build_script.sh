#!/bin/bash
set +e
LD_LIBRARY_PATH="/usr/local/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH" python packaging/wheel/relocate.py
sleep 45m
set -e
