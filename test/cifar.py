import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

print('\n\nCifar 10')
a = dset.CIFAR10(root="abc/def/ghi", download=True)

print(a[3])

# print('\n\nCifar 100')
# a = dset.CIFAR100(root="abc/def/ghi", download=True)

# print(a[3])


dataset = dset.CIFAR10(root='cifar', download=True, transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                         shuffle=True, num_workers=2)


for i, data in enumerate(dataloader, 0):
    print(data)
    if i == 10:
        break

# miter = dataloader.__iter__()
# def getBatch():
#     global miter
#     try:
#         return miter.next()
#     except StopIteration:
#         miter = dataloader.__iter__()
#         return miter.next()
    
# i=0
# while True:
#     print(i)
#     img, target = getBatch()
#     i+=1
    
