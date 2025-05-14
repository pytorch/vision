#!/bin/bash

if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null
then
    echo "NVIDIA GPU detected"
    export DOCKER_FILE_SUFFIX=gpu
else
    echo "No NVIDIA GPU detected"
    export DOCKER_FILE_SUFFIX=cpu
fi

docker-compose up --build
