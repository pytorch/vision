import warnings
import random

__all__ = ['TransformContainer', 'Compose', 'RandomTransforms', 'RandomApply',
           'RandomOrder', 'RandomChoice']


class TransformContainer(object):
    _repr_indent = 4

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        head = ["{0}({1}".format(self.__class__.__name__, self.extra_repr())]
        body = [' ' * self._repr_indent + str(t)
                for t in self.transforms]
        tail = [')']
        return '\n'.join(head + body + tail)

    def extra_repr(self):
        return ""


class Compose(TransformContainer):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomTransforms(Compose):
    """
    Note: This transform container is deprecated in favor of TransformContainer.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.RandomTransforms transform is "
                      "deprecated, please use transforms.TransformContainer instead.")
        super(RandomTransforms, self).__init__(*args, **kwargs)


class RandomApply(TransformContainer):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, img):
        if self.p < random.random():
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def extra_repr(self):
        return "p={p}".format(**self.__dict__)


class RandomOrder(TransformContainer):
    """Apply a list of transformations in a random order
    """

    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](img)
        return img


class RandomChoice(TransformContainer):
    """Apply single transformation randomly picked from a list
    """

    def __call__(self, img):
        t = random.choice(self.transforms)
        return t(img)
