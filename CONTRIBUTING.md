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


### Dependencies

Start by installing the **nightly** build of PyTorch following the [official
instructions](https://pytorch.org/get-started/locally/).

**Optionally**, install `libpng` and `libjpeg-turbo` if you want to enable
support for
native encoding / decoding of PNG and JPEG formats in
[torchvision.io](https://pytorch.org/vision/stable/io.html#image):

```bash
conda install libpng libjpeg-turbo -c pytorch
```

Note: you can use the `TORCHVISION_INCLUDE` and `TORCHVISION_LIBRARY`
environment variables to tell the build system where to find those libraries if
they are in specific locations. Take a look at
[setup.py](https://github.com/pytorch/vision/blob/main/setup.py) for more
details.

### Clone and install torchvision

```bash
git clone https://github.com/pytorch/vision.git
cd vision
python setup.py develop  # use install instead of develop if you don't care about development.
# or, for OSX
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py develop
# for C++ debugging, use DEBUG=1
# DEBUG=1 python setup.py develop
```

By default, GPU support is built if CUDA is found and `torch.cuda.is_available()` is true. It's possible to force
building GPU support by setting `FORCE_CUDA=1` environment variable, which is useful when building a docker image.

We don't officially support building from source using `pip`, but _if_ you do, you'll need to use the
`--no-build-isolation` flag.

Other development dependencies include:

```
pip install flake8 typing mypy pytest pytest-mock scipy
```

## Development Process

If you plan to modify the code or documentation, please follow the steps below:

1. Fork the repository and create your branch from `main`.
2. If you have modified the code (new feature or bug-fix), please add unit tests.
3. If you have changed APIs, update the documentation. Make sure the documentation builds.
4. Ensure the test suite passes.
5. Make sure your code passes the formatting checks (see below).

For more details about pull requests,
please read [GitHub's guides](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

If you would like to contribute a new model, please see [here](#New-architecture-or-improved-model-weights).

If you would like to contribute a new dataset, please see [here](#New-dataset).

### Code formatting and typing

#### Formatting

The torchvision code is formatted by [black](https://black.readthedocs.io/en/stable/),
and checked against pep8 compliance with [flake8](https://flake8.pycqa.org/en/latest/).
Instead of relying directly on `black` however, we rely on
[ufmt](https://github.com/omnilib/ufmt), for compatibility reasons with Facebook
internal infrastructure.

To format your code, install `ufmt` with `pip install ufmt==1.3.3 black==22.3.0 usort==1.0.2` and use e.g.:

```bash
ufmt format torchvision
```

For the vast majority of cases, this is all you should need to run. For the
formatting to be a bit faster, you can also choose to only apply `ufmt` to the
files that were edited in your PR with e.g.:

```bash
ufmt format `git diff main --name-only`
```

Similarly, you can check for `flake8` errors with `flake8 torchvision`, although
they should be fairly rare considering that most of the errors are automatically
taken care of by `ufmt` already.

##### Pre-commit hooks

For convenience and **purely optionally**, you can rely on [pre-commit
hooks](https://pre-commit.com/) which will run both `ufmt` and `flake8` prior to
every commit.

First install the `pre-commit` package with `pip install pre-commit`, and then
run `pre-commit install` at the root of the repo for the hooks to be set up -
that's it.

Feel free to read the [pre-commit docs](https://pre-commit.com/#usage) to learn
more and improve your workflow. You'll see for example that `pre-commit run
--all-files` will run both `ufmt` and `flake8` without the need for you to
commit anything, and that the `--no-verify` flag can be added to `git commit` to
temporarily deactivate the hooks.

#### Type annotations

The codebase has type annotations, please make sure to add type hints if required. We use `mypy` tool for type checking:
```bash
mypy --config-file mypy.ini
```

### Unit tests

If you have modified the code by adding a new feature or a bug-fix, please add unit tests for that. To run a specific
test:
```bash
pytest test/<test-module.py> -vvv -k <test_myfunc>
# e.g. pytest test/test_transforms.py -vvv -k test_center_crop
```

If you would like to run all tests:
```bash
pytest test -vvv
```

Tests that require internet access should be in
`test/test_internet.py`.

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
make html-noplot
```

Then open `docs/build/html/index.html` in your favorite browser.

The docs are also automatically built when you submit a PR. The job that
builds the docs is named `build_docs`. You can access the rendered docs by
clicking on that job and then going to the "Artifacts" tab.

You can clean the built docs and re-start the build from scratch by doing ``make
clean``.

#### Building the example gallery - or not

In most cases, running `make html-noplot` is enough to build the docs for your
specific use-case. The `noplot` part tells sphinx **not** to build the examples
in the [gallery](https://pytorch.org/vision/stable/auto_examples/index.html),
which saves a lot of building time.

If you need to build all the examples in the gallery, then you can use `make
html`.

You can also choose to only build a subset of the examples by using the
``EXAMPLES_PATTERN`` env variable, which accepts a regular expression. For
example ``EXAMPLES_PATTERN="transforms" make html`` will only build the examples
with "transforms" in their name.

### New architecture or improved model weights

Please refer to the guidelines in [Contributing to Torchvision - Models](https://github.com/pytorch/vision/blob/main/CONTRIBUTING_MODELS.md).

### New dataset

Please, do not send any PR with a new dataset without discussing
it in an issue as, most likely, it will not be accepted.

### Pull Request

If all previous checks (flake8, mypy, unit tests) are passing, please send a PR. Submitted PR will pass other tests on
different operating systems, python versions and hardware.

For more details about pull requests workflow,
please read [GitHub's guides](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## License

By contributing to Torchvision, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

Contributors are also required to [sign our Contributor License Agreement](https://code.facebook.com/cla).
