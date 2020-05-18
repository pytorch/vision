import unittest
from collections import defaultdict

from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
import torchvision.transforms as transforms

from sampler import PKSampler


class Tester(unittest.TestCase):

    def test_pksampler(self):
        p, k = 16, 4

        # Ensure sampler does not allow p to be greater than num_classes
        dataset = FakeData(size=100, num_classes=10, image_size=(3, 1, 1))
        targets = [target.item() for _, target in dataset]
        self.assertRaises(AssertionError, PKSampler, targets, p, k)

        # Ensure p, k constraints on batch
        dataset = FakeData(size=1000, num_classes=100, image_size=(3, 1, 1),
                           transform=transforms.ToTensor())
        targets = [target.item() for _, target in dataset]
        sampler = PKSampler(targets, p, k)
        loader = DataLoader(dataset, batch_size=p * k, sampler=sampler)

        for _, labels in loader:
            bins = defaultdict(int)
            for label in labels.tolist():
                bins[label] += 1

            # Ensure that each batch has samples from exactly p classes
            self.assertEqual(len(bins), p)

            # Ensure that there are k samples from each class
            for b in bins:
                self.assertEqual(bins[b], k)


if __name__ == '__main__':
    unittest.main()
