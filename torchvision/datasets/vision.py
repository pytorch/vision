import os
import torch.utils.data as data


class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root, transform, target_transform):
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transforms: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transforms: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""
