import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from sklearn import manifold
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from model import EmbeddingNet

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning Visualisation')

    parser.add_argument('--model-path', type=str, required=True)

    parser.add_argument('--dataset-dir', default='/tmp/fmnist/',
                        help='FashionMNIST dataset directory path')
    parser.add_argument('--eval-batch-size', default=128, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers')

    parser.add_argument('--perplexity', type=int, default=100)

    return parser.parse_args()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    embeds, labels = [], []
    dists, targets = None, None

    for data in loader:
        samples, _labels = data[0].to(device), data[1]
        out = model(samples)
        embeds.append(out)
        labels.append(_labels)

    embeds = torch.cat(embeds, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()

    return embeds, labels

def create_tsne(embedds, perplexity = 100):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity)
    return tsne.fit_transform(embedds)

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('loading model and creating data loader')

    model = EmbeddingNet()
    model.load_state_dict(torch.load(args.model_path))

    model.to(device)

    transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    test_dataset = FashionMNIST(args.dataset_dir, train=False, transform=transform, download=True)
    print('fashion mnist classes: {}'.format(test_dataset.classes))

    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                             shuffle=False,
                             num_workers=args.workers)


    print('evaluating model on test dataset')
    embeds, labels = evaluate(model, test_loader, device)

    print('computing 2D embedding')
    embeds2D = create_tsne(embeds, args.perplexity)

    data = pd.DataFrame({'x': embeds2D[:, 0], 'y': embeds2D[:, 1], 'label': labels})
    data['class'] = data['label'].map(lambda i: test_dataset.classes[i])
    
    plot = sns.scatterplot(x='x', y='y', hue='class', data=data)
    plot.figure.savefig('fashion_mnist_embedding.png')

    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)

    