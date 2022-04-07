import itertools

from torchvision.prototype import datasets


def combinations_grid(**kwargs):
    return [dict(zip(kwargs.keys(), values)) for values in itertools.product(*kwargs.values())]


# If you have implemented the mock data function for the dataset tests, you can simply copy-paste this
configs = combinations_grid(split=("train", "test"), foo=("bar", "baz"))

for config in configs:
    dataset = datasets.load("my-dataset", **config)

    num_samples = 0
    for _ in dataset:
        num_samples += 1

    print(", ".join(f"{key}={value}" for key, value in config.items()), num_samples)
