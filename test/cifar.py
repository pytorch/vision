import torch
import torchvision.datasets as dset

print('\n\nCifar 10')
a = dset.CIFAR10(root="abc/def/ghi", download=True)

print(a[3])

print('\n\nCifar 100')
a = dset.CIFAR100(root="abc/def/ghi", download=True)

print(a[3])
