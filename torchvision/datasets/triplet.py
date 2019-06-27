from .folder import DatasetFolder

import math
import torch
import torch.utils.data as data

from torch.distributions.multinomial import Multinomial
from torch.distributions.bernoulli import Bernoulli

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
    swap = Bernoulli(torch.Tensor([0.5]))
    for _ in range(num_triplets):
        pos_cls, neg_cls = get_sampled_indices(class_dist, 2)
        if swap.sample().item() == 1:
            pos_cls, neg_cls = neg_cls, pos_cls

        pos_samples, neg_samples = class_samples[pos_cls], class_samples[neg_cls]

        pos_dist = Multinomial(2, torch.Tensor([1.] * len(pos_samples)))
        neg_dist = Multinomial(1, torch.Tensor([1.] * len(neg_samples)))

        anc_idx, pos_idx = get_sampled_indices(pos_dist, 2)
        neg_idx = get_sampled_indices(neg_dist, 1)[0]
        if swap.sample().item() == 1:
            anc_idx, pos_idx = pos_idx, anc_idx

        triplet = (pos_samples[anc_idx], pos_samples[pos_idx], neg_samples[neg_idx])
        triplets.append(triplet)

    return triplets

class TripletDataset(data.IterableDataset, DatasetFolder):
    def __init__(self, root, loader, num_triplets, extensions=None, transform=None, is_valid_file=None):
        super(TripletDataset, self).__init__(root, loader,
                                             extensions=extensions,
                                             transform=transform,
                                             is_valid_file=is_valid_file)
        self.num_triplets = num_triplets

        class_samples = get_class_samples(len(self.classes), self.samples)
        self.samples = generate_triplets(class_samples, self.num_triplets)

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.num_triplets
        else:
            per_worker = int(math.ceil(self.num_triplets / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, self.num_triplets)

        return (self[i] for i in range(iter_start, iter_end))

    def __getitem__(self, index):
        #raise TypeError(f"'{self.__class__.__name__}' object is not subscriptable")
        triplet_paths = self.samples[index]
        triplet = tuple(self.loader(path) for path in triplet_paths)
        if self.transform is not None:
            triplet = tuple(self.transform(img) for img in triplet)

        return triplet
