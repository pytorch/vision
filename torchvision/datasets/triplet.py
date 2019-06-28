from .folder import DatasetFolder

import torch
import torch.utils.data as data

import math


def get_class_samples(num_classes, samples):
    """Bins samples with respect to classes

    Args:
        num_classes (int): number of classes
        samples (list[tuple]): (sample, class_idx) tuples

    Returns:
        list[list]: Bins of sample paths, binned by class_idx
    """
    class_samples = [[] for _ in range(num_classes)]
    for sample_path, class_idx in samples:
        class_samples[class_idx].append(sample_path)

    return class_samples


def generate_triplet(class_samples):
    """Generates a triplet from bins of samples

    Args:
        class_samples (list[list]): bins of samples, binned by class
        num_triplets (int): number of triplets to be generated

    Returns:
        list[tuple]: triplets of the form (anchor, positive, negative)
    """
    pos_cls, neg_cls = torch.multinomial(torch.ones(len(class_samples)), 2).tolist()
    pos_samples, neg_samples = class_samples[pos_cls], class_samples[neg_cls]

    anc_idx, pos_idx = torch.multinomial(torch.ones(len(pos_samples)), 2).tolist()
    neg_idx = torch.multinomial(torch.ones(len(neg_samples)), 1).item()

    return (pos_samples[anc_idx], pos_samples[pos_idx], neg_samples[neg_idx])


class TripletDataset(data.IterableDataset, DatasetFolder):
    """
    A dataset with samples of the form (anchor, positive, negative), where anchor and
    positive are samples of the same class, and negative is a sample of another class.
    The dataset reads samples from a directory arranged in the following manner: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions,
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        is_valid_file (callable, optional): A function that takes the path of an Image
        file and checks if the file is a valid_file. Both extensions and is_valid_file
        should not be passed.

    Attributes:
        samples (list[tuple]): List of (anchor, positive, negative) triplets
    """

    def __init__(self, root, loader, num_triplets, extensions=None, transform=None, is_valid_file=None):
        super(TripletDataset, self).__init__(root, loader,
                                             extensions=extensions,
                                             transform=transform,
                                             is_valid_file=is_valid_file)
        self.num_triplets = num_triplets
        self.class_samples = get_class_samples(len(self.classes), self.samples)

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.num_triplets
        else:
            per_worker = int(math.ceil(self.num_triplets / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, self.num_triplets)

        return (self.load(generate_triplet(self.class_samples)) for _ in range(iter_start, iter_end))

    def load(self, triplet_paths):
        triplet = tuple(self.loader(path) for path in triplet_paths)
        if self.transform is not None:
            triplet = tuple(self.transform(img) for img in triplet)

        return triplet
