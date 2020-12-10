# Contributing to Torchvision

We want to make contributing to this project as easy and transparent as possible.

## TL;DR

We appreciate all contributions. If you are interested in contributing to Torchvision, there are many ways to help out. 
Your contributions may fall into the following categories:

- It helps the project if you could 
    - Report issues you're facing
    - Give a :+1: on issues that others reported and that are relevant to you 

- Answering queries on the issue tracker, investigating bugs are very valuable contributions to the project.

- You would like to improve the documentation. This is no less important than improving the library itself! 
If you find a typo in the documentation, do not hesitate to submit a GitHub pull request.

- If you would like to fix a bug
    - please pick one from the [list of open issues labelled as "help wanted"](https://github.com/pytorch/vision/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)
    - comment on the issue that you want to work on this issue
    - send a PR with your fix, see below. 

- If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## Development installation

### Install PyTorch Nightly 

```bash
conda install pytorch -c pytorch-nightly
# or with pip (see https://pytorch.org/get-started/locally/)
# pip install numpy
# pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
```

### Install Torchvision

```bash
git clone https://github.com/pytorch/vision.git
cd vision
python setup.py install
# or, for OSX
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
# for C++ debugging, please use DEBUG=1
# DEBUG=1 python setup.py install
pip install flake8 typing mypy pytest scipy
```
You may also have to install `libpng-dev` and `libjpeg-turbo8-dev` libraries:
```bash
conda install libpng jpeg
```

## Development Process

If you plan to modify the code or documentation, please follow the steps below:

1. Fork the repository and create your branch from `master`.
2. If you have modified the code (new feature or bug-fix), please add unit tests.
3. If you have changed APIs, update the documentation. Make sure the documentation builds.
4. Ensure the test suite passes.
5. Make sure your code passes `flake8` formatting check.

For more details about pull requests, 
please read [GitHub's guides](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request). 

If you would like to contribute a new model, please see [here](#New-model).

If you would like to contribute a new dataset, please see [here](#New-dataset). 

### Code formatting and typing

New code should be compatible with Python 3.X versions and be compliant with PEP8. To check the codebase, please run
```bash
flake8 --config=setup.cfg .
```

The codebase has type annotations, please make sure to add type hints if required. We use `mypy` tool for type checking:
```bash
mypy --config-file mypy.ini
```

### Unit tests

If you have modified the code by adding a new feature or a bug-fix, please add unit tests for that. To run a specific 
test: 
```bash
pytest test/<test-module.py> -vvv -k <test_myfunc>
# e.g. pytest test/test_transforms.py -vvv -k test_crop
```

If you would like to run all tests:
```bash
pytest test -vvv
``` 

### Documentation

Torchvision uses [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
for formatting docstrings. Length of line inside docstrings block must be limited to 120 characters.

Please, follow the instructions to build and deploy the documentation locally.

#### Install requirements

```bash
cd docs
pip install -r requirements.txt
```

#### Build

```bash
cd docs
make html
```

#### Local deployment

Please, use python 3.X for the command below:
```bash
cd docs/build/html
python -m http.server <port>
# e.g. python -m http.server 1234
```
Then open the browser at `0.0.0.0:<port>` (e.g. `0.0.0.0:1234`)

### New model

More details on how to add a new model will be provided later. Please, do not send any PR with a new model without discussing 
it in an issue as, most likely, it will not be accepted.
 
### New dataset

More details on how to add a new dataset will be provided later. Please, do not send any PR with a new dataset without discussing 
it in an issue as, most likely, it will not be accepted.

### Pull Request

If all previous checks (flake8, mypy, unit tests) are passing, please send a PR. Submitted PR will pass other tests on 
different operation systems, python versions and hardwares.

For more details about pull requests workflow, 
please read [GitHub's guides](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## License

By contributing to Torchvision, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
