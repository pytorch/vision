import os

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
            i += 1
            avg_loss = running_loss / print_freq
            avg_trip = 100.0 * running_frac_pos_triplets / print_freq
            print(f'[{epoch:d}, {i:d}] | loss: {avg_loss:.4f} | % avg hard triplets: {avg_trip:.2f}%')
            running_loss = 0
            running_frac_pos_triplets = 0


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    embeds, labels = None, None
    dist_fn = PairwiseDistance(2)
    dists, targets = None, None

    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    for data in loader:
        samples, _labels = data[0].to(device), data[1]
        out = model(samples)
        embeds = torch.cat((embeds, out), dim=0) if embeds is not None else out
        labels = torch.cat((labels, _labels), dim=0) if labels is not None else _labels

    dists = torch.cdist(embeds, embeds)

    labels = labels.unsqueeze(0)
    targets = labels == labels.t()

    mask = torch.ones(dists.size()).triu() - torch.eye(dists.size(0))
    dists = dists[mask == 1]
    targets = targets[mask == 1]

    best_thresh = 0.01
    best_correct = 0
    for thresh in torch.arange(0.0, 1.51, 0.01):
        predictions = dists <= thresh.to(device)
        correct = torch.sum(predictions == targets.to(device)).item()
        if correct > best_correct:
            best_thresh = thresh
            best_correct = correct

    accuracy = 100.0 * best_correct / dists.size(0)
    print(f'accuracy: {accuracy:.3f}%, threshold: {best_thresh:.2f}')


def save(model, epoch, save_dir, file_name):
    file_name = 'epoch_' + str(epoch) + '__' + file_name
    save_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), save_path)


def main():
    p = 8
    k = 8
    batch_size = p * k
    print_freq = 200
    epochs = 5

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


    targets = train_dataset.targets.tolist()
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=PKSampler(targets, p, k),
                              num_workers=4)

    for epoch in range(1, epochs + 1):
        print('Training...')
        train_epoch(model, optimizer, criterion, train_loader, device, epoch, print_freq)

        print('Evaluating...')
        evaluate(model, test_dataset, device)

        print('Saving...')
        save(model, epoch, './saved_models/', 'testnet.pth')


if __name__ == '__main__':
    main()
