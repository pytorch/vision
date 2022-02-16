# Building torchvision packages for release

TorchVision release packages are build by invoking `build_wheel.sh` and `build_conda.sh` for all OSes, compute and python version permutation

OS/Python/CUDA matrix is defined in https://github.com/pytorch/vision/blob/main/.circleci/regenerate.py
