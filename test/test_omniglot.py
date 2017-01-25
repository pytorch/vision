import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

print('Omniglot')
a = dset.OMNIGLOT("../data", download=True,transform=transforms.Compose([transforms.FilenameToPILImage(),transforms.ToTensor()]))

print(a.idx_classes)
print(a[3])
# print('\n\nCifar 100')
# a = dset.CIFAR100(root="abc/def/ghi", download=True)

# print(a[3])
