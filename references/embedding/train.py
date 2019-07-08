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


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    p = args.labels_per_batch
    k = args.samples_per_label
    batch_size = p * k

    model = EmbeddingNet()
    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    model.to(device)

    criterion = TripletMarginLoss(margin=args.margin)
    optimizer = Adam(model.parameters(), lr=args.lr)

    transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    train_dataset = FashionMNIST(args.train_data, train=True, transform=transform, download=True)
    test_dataset = FashionMNIST(args.test_data, train=False, transform=transform, download=True)

    targets = train_dataset.targets.tolist()
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=PKSampler(targets, p, k),
                              num_workers=4)

    for epoch in range(1, args.epochs + 1):
        print('Training...')
        train_epoch(model, optimizer, criterion, train_loader, device, epoch, args.print_freq)

        print('Evaluating...')
        evaluate(model, test_dataset, device)

        print('Saving...')
        save(model, epoch, args.save_dir, 'ckpt.pth')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning')

    parser.add_argument('--train-data', default='/tmp/pyemb/train/',
                        help='FashionMNIST train dataset path')
    parser.add_argument('--test-data', default='/tmp/pyemb/test/',
                        help='FashionMNIST test dataset path')
    parser.add_argument('-p', '--labels-per-batch', default=8, type=int,
                        help='Number of unique labels/classes per batch')
    parser.add_argument('-k', '--samples-per-label', default=8, type=int,
                        help='Number of samples per label in a batch')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='Number of training epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--margin', default=0.2, type=float, help='Triplet loss margin')
    parser.add_argument('--print-freq', default=20, type=int, help='Print frequency')
    parser.add_argument('--save-dir', default='.', help='Model save directory')
    parser.add_argument('--resume', default='', help='Resume from checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
