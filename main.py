from torchvision.prototype import datasets
import tqdm

datasets.home("~/datasets")

dataset = datasets.load("caltech101", decoder=None)

for sample in tqdm.tqdm(dataset):
    pass
