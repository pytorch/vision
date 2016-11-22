import argparse
import os
from timeit import default_timer as timer
from tqdm import tqdm
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='PATH', required=True,
                    help='path to dataset')
parser.add_argument('--nThreads', '-j', default=2, type=int, metavar='N',
                    help='number of data loading threads (default: 2)')
parser.add_argument('--batchSize', '-b', default=256, type=int, metavar='N',
                    help='mini-batch size (1 = pure stochastic) Default: 256')


if __name__ == "__main__":
    args = parser.parse_args()


    # Data loading code
    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                             std = [ 0.229, 0.224, 0.225 ]),
    ])

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    train = datasets.ImageFolder(traindir, transform)
    val = datasets.ImageFolder(valdir, transform)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=args.batchSize, shuffle=True, num_workers=args.nThreads)
    train_iter = iter(train_loader)

    start_time = timer()
    batch_count = 100 * args.nThreads
    for i in tqdm(xrange(batch_count)):
        batch = next(train_iter)
    end_time = timer()
    print("Performance: {dataset:.0f} minutes/dataset, {batch:.2f} secs/batch, {image:.2f} ms/image".format(
        dataset=(end_time - start_time) * len(train_loader) / (batch_count * args.batchSize) / 60.0,
        batch=(end_time - start_time) / float(batch_count),
        image=(end_time - start_time) / (batch_count * args.batchSize) * 1.0e+3))

