import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import random


def create_groups(groups, k):
    """Bins sample indices with respect to groups, remove bins with less than k samples

    Args:
        groups (list[int]): where ith index stores ith sample's group id

    Returns:
        defaultdict[list]: Bins of sample indices, binned by group_idx
    """
    group_samples = defaultdict(list)
    for sample_idx, group_idx in enumerate(groups):
        group_samples[group_idx].append(sample_idx)

    keys_to_remove = []
    for key in group_samples:
        if len(group_samples[key]) < k:
            keys_to_remove.append(key)
            continue

    for key in keys_to_remove:
        group_samples.pop(key)

    return group_samples


class PKSampler(Sampler):
    """
    Randomly samples from a dataset  while ensuring that each batch (of size p * k)
    includes samples from exactly p labels, with k samples for each label.

    Args:
        groups (list[int]): List where the ith entry is the group_id/label of the ith sample in the dataset.
        p (int): Number of labels/groups to be sampled from in a batch
        k (int): Number of samples for each label/group in a batch
    """

    def __init__(self, groups, p, k):
        self.p = p
        self.k = k
        self.groups = create_groups(groups, self.k)

        # Ensures there are enough classes to sample from
        assert len(self.groups) >= p

    def __iter__(self):
        # Shuffle samples within groups
        for key in self.groups:
            random.shuffle(self.groups[key])

        # Keep track of the number of samples left for each group
        group_samples_remaining = {}
        for key in self.groups:
            group_samples_remaining[key] = len(self.groups[key])

        while len(group_samples_remaining) > self.p:
            # Select p groups at random from valid/remaining groups
            group_ids = list(group_samples_remaining.keys())
            selected_group_idxs = torch.multinomial(torch.ones(len(group_ids)), self.p).tolist()
            for i in selected_group_idxs:
                group_id = group_ids[i]
                group = self.groups[group_id]
                for _ in range(self.k):
                    # No need to pick samples at random since group samples are shuffled
                    sample_idx = len(group) - group_samples_remaining[group_id]
                    yield group[sample_idx]
                    group_samples_remaining[group_id] -= 1

                # Don't sample from group if it has less than k samples remaining
                if group_samples_remaining[group_id] < self.k:
                    group_samples_remaining.pop(group_id)
