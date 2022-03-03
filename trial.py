from torchvision.prototype import datasets

dataset = datasets.load("stanford-cars")

for sample in dataset:
    if sample["image_path"][-9:] != sample["index1"]:
        print("Use IterKeyZipper instead of Zipper")

# Or you can also inspect the sample in a debugger