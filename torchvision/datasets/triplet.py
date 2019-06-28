import torch
import torch.utils.data as data

import math
from collections import defaultdict


def get_class_samples(samples):
    """Bins samples with respect to classes

    Args:
        num_classes (int): number of classes
        samples (list[tuple]): (sample, class_idx) tuples

    Returns:
        list[list]: Bins of sample paths, binned by class_idx
    """
    class_samples = defaultdict(list)
    for sample_path, class_idx in samples:
        class_samples[class_idx].append(sample_path)

    return class_samples


def generate_triplet(class_samples):
    """Generates a triplet from bins of samples

    Args:
        class_samples (list[list]): bins of samples, binned by class

    Returns:
        tuple(str): triplet of the form (anchor, positive, negative)
    """
    pos_cls, neg_cls = torch.multinomial(torch.ones(len(class_samples)), 2).tolist()
    pos_samples, neg_samples = class_samples[pos_cls], class_samples[neg_cls]

    anc_idx, pos_idx = torch.multinomial(torch.ones(len(pos_samples)), 2).tolist()
    neg_idx = torch.multinomial(torch.ones(len(neg_samples)), 1).item()

    return (pos_samples[anc_idx], pos_samples[pos_idx], neg_samples[neg_idx])


class TripletDataset(data.IterableDataset):
    """
    A dataset with samples of the form (anchor, positive, negative), where anchor and
    positive are samples of the same class, and negative is a sample of another class.
    TripletDataset reads fram Dataset `dset` where `dset[i]` returns (sample_path, class_idx).

    Args:
        dset (Dataset): Dataset object where __getitem__ returns (sample_path, class_idx) tuple.
        loader (callable): A function to load a sample given its path.
        num_triplets (int): Number of triplets to generate before raising StopIteration.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.

    Attributes:
        samples (list[tuple]): List of (anchor, positive, negative) triplets
    """

    def __init__(self, dset, loader, num_triplets, transform=None):
        super(TripletDataset, self).__init__()
        self.loader = loader
        self.transform = transform
        self.num_triplets = num_triplets
        self.class_samples = get_class_samples(dset)

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None:
            num_iters = self.num_triplets
        else:
            num_iters = int(math.ceil(self.num_triplets / float(worker_info.num_workers)))
            if worker_info.id == worker_info.num_workers - 1:
                num_iters = self.num_triplets - num_iters * worker_info.id

        return (self.load(generate_triplet(self.class_samples)) for _ in range(num_iters))

    def load(self, triplet_paths):
        triplet = tuple(self.loader(path) for path in triplet_paths)
        if self.transform is not None:
            triplet = tuple(self.transform(img) for img in triplet)

        return triplet
