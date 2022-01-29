# Contributing to Torchvision - Models

- [New Model Architectures - Overview](#new-model-architectures---overview)

- [New Weights for Existing Model Architectures](#new-weights-for-existing-model-architectures)

## New Model Architectures - Overview

For someone who would be interested in adding a model architecture, it is also expected to train the model, so here are a few important considerations:

- Training big models requires lots of resources and the cost quickly adds up

- Reproducing models is fun but also risky as you might not always get the results reported on the paper. It might require a huge amount of effort to close the gap

- The contribution might not get merged if we significantly lack in terms of accuracy, speed etc

- Including new models in TorchVision might not be the best approach, so other options such as releasing the model through to [Pytorch Hub](https://pytorch.org/hub/) should be considered

So, before starting any work and submitting a PR there are a few critical things that need to be taken into account in order to make sure the planned contribution is within the context of TorchVision, and the requirements and expectations are discussed beforehand. If this step is skipped and a PR is submitted without prior discussion it will almost certainly be rejected.

### 1. Preparation work

- Start by looking into this [issue](https://github.com/pytorch/vision/issues/2707) in order to have an idea of the models that are being considered, express your willingness to add a new model and discuss with the community whether or not this model should be included in TorchVision. It is very important at this stage to make sure that there is an agreement on the value of having this model in TorchVision and there is no one else already working on it.

- If the decision is to include the new model, then please create a new ticket which will be used for all design and implementation discussions prior to the PR. One of the TorchVision maintainers will reach out at this stage and this will be your POC from this point onwards in order to provide support, guidance and regular feedback.

### 2.  Implement the model

Please take a look at existing models in TorchVision to get familiar with the idioms. Also please look at recent contributions for new models. If in doubt about any design decisions you can ask for feedback on the issue created in step 1.  Example of things to take into account:

- The implementation should be as close as possible to the canonical implementation/paper
- The PR must include the code implementation, documentation and tests
- It should also extend the existing reference scripts used to train the model
- The weights need to reproduce closely the results of the paper in terms of accuracy, even though the final weights to be deployed will be those trained by the TorchVision maintainers
- The PR description should include commands/configuration used to train the model, so that the TorchVision maintainers can easily run them to verify the implementation and generate the final model to be released
- Make sure we re-use existing components as much as possible (inheritance)
- New primitives (transforms, losses, etc) can be added if necessary, but the final location will be determined after discussion with the dedicated maintainer
- Please take a look at the detailed [implementation and documentation guidelines](https://github.com/pytorch/vision/issues/5319) for a fine grain list of things not to be missed

### 3. Train the model with reference scripts

To validate the new model against the common benchmark, as well as to generate pre-trained weights, you must use TorchVision’s reference scripts to train the model.

Make sure all logs and a final (or best) checkpoint are saved, because it is expected that a submission shows that a model has been successfully trained  and the results are in line with the original paper/repository. This will allow the reviewers to quickly check the validity of the submission, but please note that the final model to be released will be re-trained by the maintainers in order to verify reproducibility,  ensure that the changes occurred during the PR review did not introduce any bugs, and to avoid moving around a large amount of data (including all checkpoints and logs).

### 4. Submit a PR

Submit a PR and tag the assigned maintainer. This PR should:

- Link the original ticket
- Provide a link for the original paper and the original repository if available
- Highlight the important test metrics and how they compare to the original paper
- Highlight any design choices that deviate from the original paper/implementation and rationale for these choices

## New Weights for Existing Model Architectures

The process of improving existing models, for instance improving accuracy by retraining the model with a different set of hyperparameters or augmentations, is the following:

1. Open a ticket and discuss with the community and maintainers whether this improvement should be added to TorchVision. Note that to add new weights the improvement should be significant.

2. Train the model using TorchVision reference scripts. You can add new primitives (transforms, losses, etc) when necessary, but the final location will be determined after discussion with the dedicated maintainer.

3. Open a PR with the new weights, together with the training logs and the checkpoint chosen so the reviewers can verify the submission.  Details on how the model was trained, i.e., the training command using the reference scripts, should be included in the PR.

4. The PR reviewers should replicate the results on their side to verify the submission and if all goes well the new weights should be ready to be released!
=======
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
python setup.py develop
# or, for OSX
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py develop
# for C++ debugging, please use DEBUG=1
# DEBUG=1 python setup.py develop
pip install flake8 typing mypy pytest pytest-mock scipy
```
You may also have to install `libpng-dev` and `libjpeg-turbo8-dev` libraries:
```bash
conda install libpng jpeg
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

If you would like to contribute a new model, please see [here](#New-model).

If you would like to contribute a new dataset, please see [here](#New-dataset). 

### Code formatting and typing

#### Formatting

The torchvision code is formatted by [black](https://black.readthedocs.io/en/stable/),
and checked against pep8 compliance with [flake8](https://flake8.pycqa.org/en/latest/).
Instead of relying directly on `black` however, we rely on
[ufmt](https://github.com/omnilib/ufmt), for compatibility reasons with Facebook
internal infrastructure.

To format your code, install `ufmt` with `pip install ufmt` and use e.g.:

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
make html
```

Then open `docs/build/html/index.html` in your favorite browser.

The docs are also automatically built when you submit a PR. The job that
builds the docs is named `build_docs`. You can access the rendered docs by
clicking on that job and then going to the "Artifacts" tab.

You can clean the built docs and re-start the build from scratch by doing ``make
clean``.

#### Building the example gallery - or not

When you run ``make html`` for the first time, all the examples in the gallery
will be built. Subsequent builds should be faster, and will only build the
examples that have been modified.

You can run ``make html-noplot`` to not build the examples at all. This is
useful after a ``make clean`` to do some quick checks that are not related to
the examples.

You can also choose to only build a subset of the examples by using the
``EXAMPLES_PATTERN`` env variable, which accepts a regular expression. For
example ``EXAMPLES_PATTERN="transforms" make html`` will only build the examples
with "transforms" in their name.

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
