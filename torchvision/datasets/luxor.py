import torch
import torchvision

from ipsc_luxor import IPSCLuxor

def main():
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))])

    trainset = IPSCLuxor(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True)

    testset = IPSCLuxor(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False)

    classes = trainset.class_to_idx.keys()
    print(classes)

if __name__ == "__main__":
    main()
