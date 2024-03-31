# Similarity Learning Using Triplet Loss #

In this reference, we use triplet loss to learn embeddings which can be used to differentiate images. This learning technique was popularized by [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) and has been quite effective in learning embeddings to differentiate between faces.

This reference can be directly applied to the following use cases:

* You have an unknown number of classes and would like to train a model to learn how to differentiate between them.
* You want to train a model to learn a distance-based metric between samples. For example, learning a distance-based similarity measure between faces.

### Training ###
By default, the training script trains ResNet50 on the FashionMNIST Dataset to learn image embeddings which can be used to differentiate between images by measuring the euclidean distance between embeddings. This can be changed as per your requirements.

Image embeddings of the same class should be 'close' to each other, while image embeddings between different classes should be 'far' away.

To run the training script:

```bash
python train.py -h    # Lists all optional arguments
python train.py 			# Runs training script with default args
```

Running the training script as is should yield 97% accuracy on the FMNIST test set within 10 epochs.

### Loss ###
`TripletMarginLoss` is a loss function which takes in a triplet of samples. A valid triplet has an:

1. Anchor: a sample from the dataset
2. Positive: another sample with the same label/group as the anchor (Generally, positive != anchor)
3. Negative: a sample with a different label/group from the anchor

`TripletMarginLoss` (refer to `loss.py`) does the following:

```python
loss = max(dist(anchor, positive) - dist(anchor, negative) + margin, 0)
```
Where `dist` is a distance function. Minimizing this function effectively leads to minimizing `dist(anchor, positive)` and maximizing `dist(anchor, negative)`.

The FaceNet paper describe this loss in more detail.

### Sampler ###

In order to generate valid triplets from a batch of samples, we need to make sure that each batch has multiple samples with the same label. We do this using `PKSampler` (refer to `sampler.py`), which ensures that each batch of size `p * k` will have samples from exactly `p` classes and `k` samples per class.

### Triplet Mining ###

`TripletMarginLoss` currently supports the following mining techniques:

* `batch_all`: Generates all possible triplets from a batch and excludes the triplets which are 'easy' (which have `loss = 0`) before passing it through the loss function.
* `batch_hard`: For every anchor, `batch_hard` creates a triplet with the 'hardest' positive (farthest positive) and negative (closest negative).

These mining strategies usually speed up training.

This [webpage](https://omoindrot.github.io/triplet-loss) describes the sampling and mining strategies in more detail. 
