# Building torchvision packages for release

TorchVision release packages are built by using `build_wheel.sh` and `build_conda.sh` for all permutations of
supported operating systems, compute platforms and python versions.

OS/Python/Compute matrix is defined in https://github.com/pytorch/vision/blob/main/.circleci/regenerate.py
