import argparse
import os
from timeit import default_timer as timer

import torch
import torch.utils.data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.model_zoo import tqdm


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--data", metavar="PATH", required=True, help="path to dataset")
parser.add_argument(
    "--nThreads", "-j", default=2, type=int, metavar="N", help="number of data loading threads (default: 2)"
)
parser.add_argument(
    "--batchSize", "-b", default=256, type=int, metavar="N", help="mini-batch size (1 = pure stochastic) Default: 256"
)
parser.add_argument("--accimage", action="store_true", help="use accimage")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.accimage:
        torchvision.set_image_backend("accimage")
    print(f"Using {torchvision.get_image_backend()}")

    # Data loading code
    transform = transforms.Compose(
        [
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    train = datasets.ImageFolder(traindir, transform)
    val = datasets.ImageFolder(valdir, transform)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=args.batchSize, shuffle=True, num_workers=args.nThreads
    )
    train_iter = iter(train_loader)

    start_time = timer()
    batch_count = 20 * args.nThreads
    with tqdm(total=batch_count) as pbar:
        for _ in tqdm(range(batch_count)):
            pbar.update(1)
            batch = next(train_iter)
    end_time = timer()
    print(
        "Performance: {dataset:.0f} minutes/dataset, {batch:.1f} ms/batch,"
        " {image:.2f} ms/image {rate:.0f} images/sec".format(
            dataset=(end_time - start_time) * (float(len(train_loader)) / batch_count / 60.0),
            batch=(end_time - start_time) / float(batch_count) * 1.0e3,
            image=(end_time - start_time) / (batch_count * args.batchSize) * 1.0e3,
            rate=(batch_count * args.batchSize) / (end_time - start_time),
        )
    )
