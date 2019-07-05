import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.nn.modules.distance import PairwiseDistance

import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from loss import TripletMarginLoss
from sampler import PKSampler
from model import EmbeddingNet


def train_epoch(model, optimizer, criterion, data_loader, device, epoch, print_freq):
    model.train()
    running_loss = 0
    running_frac_pos_triplets = 0
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        samples, targets = data[0].to(device), data[1].to(device)

        embeddings = model(samples)

        loss, frac_pos_triplets = criterion(embeddings, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_frac_pos_triplets += float(frac_pos_triplets)

        if i % print_freq == print_freq - 1:
            epoch_percent = 100.0 * i / len(data_loader)
            avg_loss = running_loss / print_freq
            avg_trip = 100.0 * running_frac_pos_triplets / print_freq
            print(f'[{epoch:d}, {epoch_percent:.0f}%] | loss: {avg_loss:.4f} | % avg hard trips: {avg_trip:.2f}%')


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    dist_fn = PairwiseDistance(2)
    dists, targets = None, None

    loader_1 = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, data in enumerate(loader_1):
        anchor_img, anchor_label = data[0].to(device), data[1].item()
        anchor_out = model(anchor_img)
        del anchor_img

        dset_subset = Subset(dataset, list(range(i + 1, len(dataset))))
        loader_2 = DataLoader(dset_subset, batch_size=64, shuffle=False)
        for data in loader_2:
            compare_imgs, compare_labels = data[0].to(device), data[1].to(device)
            compare_out = model(compare_imgs)

            batch_dist = dist_fn(anchor_out, compare_out).cpu()
            dists = torch.cat((dists, batch_dist)) if dists is not None else batch_dist

            batch_labels = (compare_labels == anchor_label).cpu()
            targets = torch.cat((targets, batch_labels)) if targets is not None else batch_labels

    best_thresh = 0.01
    best_correct = 0
    for thresh in torch.arange(0.0, 1.51, 0.01):
        predictions = dists <= thresh
        correct = torch.sum(predictions == targets)
        if correct > best_correct:
            best_thresh = thresh
            best_correct = correct

    accuracy = best_correct / dists.size(0)
    print(f'accuracy: {accuracy}%, threshold: {best_thresh}')


def save(model, epoch, save_dir, file_name):
    pass


def tuple_transform(tpl):
    sample1, sample2 = tpl
    img1, img2 = sample1[0], sample2[0]
    target = 1. if sample1[1] == sample2[1] else 0.
    return (img1, img2, target)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = EmbeddingNet()
    model.to(device)

    criterion = TripletMarginLoss(margin=0.2)
    optimizer = Adam(model.parameters(), lr=0.0001)

    transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    train_dataset = FashionMNIST('./datasets/train', train=True, transform=transform, download=True)
    test_dataset = FashionMNIST('./datasets/test', train=False, transform=transform, download=True)

    p = 16
    k = 4
    print_freq = 300
    epochs = 10

    train_sampler = PKSampler(train_dataset.targets, p, k)
    train_loader = DataLoader(train_dataset, batch_size=p * k, sampler=train_sampler)

    print(len(test_dataset))
    for epoch in range(epochs):
        #train_epoch(model, optimizer, criterion, train_loader, device, epoch, print_freq)
        evaluate(model, test_dataset, device)


if __name__ == '__main__':
    main()
