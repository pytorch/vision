import torch
import torch.utils.data as data

import math
from collections import defaultdict


def create_groups(groups):
    """Bins sample indices with respect to groups

    Args:
        groups (list[int]): where ith index stores ith sample's group id

    Returns:
        defaultdict[list]: Bins of sample indices, binned by group_idx
    """
    group_samples = defaultdict(list)
    for sample_idx, group_idx in enumerate(groups):
        group_samples[group_idx].append(sample_idx)

    return group_samples


class TripletDataset(data.IterableDataset):
    """
    A dataset with samples of the form (anchor, positive, negative), where anchor and
    positive are samples of the same class, and negative is a sample of another class.
    TripletDataset reads fram Dataset `dset` where `dset[i]` returns (sample_path, class_idx).

    Args:
        dset (Dataset): Dataset object where __getitem__ returns (sample_path, class_idx) tuple.
        num_triplets (int): Number of triplets to generate before raising StopIteration.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
    """

    def __init__(self, dataset, num_triplets, groups, transform=None):
        super(TripletDataset, self).__init__()
        assert len(dataset) == len(groups)
        self.dset = dataset
        self.num_triplets = num_triplets
        self.transform = transform
        self.groups = create_groups(groups)

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None:
            num_iters = self.num_triplets
        else:
            num_iters = int(math.ceil(self.num_triplets / float(worker_info.num_workers)))
            if worker_info.id == worker_info.num_workers - 1:
                num_iters = self.num_triplets - num_iters * worker_info.id

        return (self.load(self.generate_triplet()) for _ in range(num_iters))

    def generate_triplet(self):
        """Generates a triplet from bins of samples

        Returns:
            tuple(int): triplet of the form (anchor, positive, negative)
        """
        pos_cls, neg_cls = torch.multinomial(torch.ones(len(self.groups)), 2).tolist()
        pos_samples, neg_samples = self.groups[pos_cls], self.groups[neg_cls]

        anc_idx, pos_idx = torch.multinomial(torch.ones(len(pos_samples)), 2).tolist()
        neg_idx = torch.multinomial(torch.ones(len(neg_samples)), 1).item()

        return (pos_samples[anc_idx], pos_samples[pos_idx], neg_samples[neg_idx])

    def load(self, triplet_idxs):
        anc_idx, pos_idx, neg_idx = triplet_idxs
        triplet = (self.dset[anc_idx], self.dset[pos_idx], self.dset[neg_idx])
        if self.transform is not None:
            triplet = self.transform(triplet)

        return triplet
