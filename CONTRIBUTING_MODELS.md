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

- Start by looking into this [issue](https://github.com/pytorch/vision/issues/2707) in order to have an idea of the models that are being considered, express your willingness to add a new model and discuss with the community whether this model should be included in TorchVision. It is very important at this stage to make sure that there is an agreement on the value of having this model in TorchVision and there is no one else already working on it.

- If the decision is to include the new model, then please create a new ticket which will be used for all design and implementation discussions prior to the PR. One of the TorchVision maintainers will reach out at this stage and this will be your POC from this point onwards in order to provide support, guidance and regular feedback.

### 2.  Implement the model

Please take a look at existing models in TorchVision to get familiar with the idioms. Also, please look at recent contributions for new models. If in doubt about any design decisions you can ask for feedback on the issue created in step 1.  Example of things to take into account:

- The implementation should be as close as possible to the canonical implementation/paper
- The PR must include the code implementation, documentation and tests
- It should also extend the existing reference scripts used to train the model
- The weights need to reproduce closely the results of the paper in terms of accuracy, even though the final weights to be deployed will be those trained by the TorchVision maintainers
- The PR description should include commands/configuration used to train the model, so that the TorchVision maintainers can easily run them to verify the implementation and generate the final model to be released
- Make sure we re-use existing components as much as possible (inheritance)
- New primitives (transforms, losses, etc.) can be added if necessary, but the final location will be determined after discussion with the dedicated maintainer
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
