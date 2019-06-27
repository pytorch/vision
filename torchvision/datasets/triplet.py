from .folder import DatasetFolder

import torch
import torch.utils.data as data

from torch.distributions.multinomial import Multinomial

def get_class_samples(num_classes, samples):
    class_samples = [[] for _ in range(num_classes)]
    for sample_path, class_idx in samples:
        class_samples[class_idx].append(sample_path)

    return class_samples

def get_sampled_indices(dist, num_samples):
    sample = dist.sample().nonzero().view(-1)
    while len(sample) != num_samples:
        sample = dist.sample().nonzero().view(-1)

    return sample.tolist()

def generate_triplets(class_samples, num_triplets):
    triplets = []
    class_dist = Multinomial(2, torch.Tensor([1.] * len(class_samples)))
    for _ in range(num_triplets):
        pos_cls, neg_cls = get_sampled_indices(class_dist, 2)
        pos_samples, neg_samples = class_samples[pos_cls], class_samples[neg_cls]

        pos_dist = Multinomial(2, torch.Tensor([1.] * len(pos_samples)))
        neg_dist = Multinomial(1, torch.Tensor([1.] * len(neg_samples)))

        anc_idx, pos_idx = get_sampled_indices(pos_dist, 2)
        neg_idx = get_sampled_indices(neg_dist, 1)
        triplet = (pos_samples[anc_idx], pos_samples[pos_idx], neg_samples[neg_idx])

        triplets.append(triplet)

    return triplets

class TripletDataset(data.IterableDataset, DatasetFolder):
    def __init__(self, root, loader, num_triplets, extensions=None, transform=None, target_transform=None, is_valid_file=None):
        super(TripletDataset, self).__init__(root, loader,
                                             extensions=extensions,
                                             transform=transform,
                                             target_transform=target_transform,
                                             is_valid_file=is_valid_file)
        self.num_triplets = num_triplets

        class_samples = get_class_samples(len(self.classes), self.samples)
        self.samples = generate_triplets(class_samples, self.num_triplets)

    def __getitem__(self, index):
        raise TypeError(f"'{self.__class__.__name__}' object is not subscriptable")
