import torch
from torchvision.models.detection import _utils
import unittest


class Tester(unittest.TestCase):
    def test_balanced_positive_negative_sampler(self):
        sampler = _utils.BalancedPositiveNegativeSampler(4, 0.25)
        # keep all 6 negatives first, then add 3 positives, last two are ignore
        matched_idxs = [torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1])]
        pos, neg = sampler(matched_idxs)
        # we know the number of elements that should be sampled for the positive (1)
        # and the negative (3), and their location. Let's make sure that they are
        # there
        self.assertEqual(pos[0].sum(), 1)
        self.assertEqual(pos[0][6:9].sum(), 1)
        self.assertEqual(neg[0].sum(), 3)
        self.assertEqual(neg[0][0:6].sum(), 3)


if __name__ == '__main__':
    unittest.main()
