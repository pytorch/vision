import random

__all__ = ['Compose', 'RandomTransforms', 'RandomApply', 'RandomOrder',
           'RandomChoice']


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    _repr_indent = 4

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        head = ["{0}({1}".format(self.__class__.__name__, self.extra_repr())]
        body = [' ' * self._repr_indent + str(t)
                for t in self.transforms]
        tail = [')']
        return '\n'.join(head + body + tail)

    def extra_repr(self):
        return ""


class RandomTransforms(Compose):
    """Base class for a list of transformations with randomness

    Args:
        transforms (list or tuple): list of transformations
    """
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class RandomApply(RandomTransforms):
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


class RandomOrder(RandomTransforms):
    """Apply a list of transformations in a random order
    """

    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](img)
        return img


class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list
    """

    def __call__(self, img):
        t = random.choice(self.transforms)
        return t(img)
