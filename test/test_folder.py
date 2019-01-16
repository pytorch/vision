import unittest

import os

from torchvision.datasets import ImageFolder
from torch._utils_internal import get_file_path_2


def mock_transform(return_value, arg_list):
    def mock(arg):
        arg_list.append(arg)
        return return_value
    return mock


class Tester(unittest.TestCase):
    root = get_file_path_2('test/assets/dataset/')
    classes = ['a', 'b']
    class_a_images = [get_file_path_2(os.path.join('test/assets/dataset/a/', path))
                      for path in ['a1.png', 'a2.png', 'a3.png']]
    class_b_images = [get_file_path_2(os.path.join('test/assets/dataset/b/', path))
                      for path in ['b1.png', 'b2.png', 'b3.png', 'b4.png']]

    def test_image_folder(self):
        dataset = ImageFolder(Tester.root, loader=lambda x: x)
        self.assertEqual(sorted(Tester.classes), sorted(dataset.classes))
        for cls in Tester.classes:
            self.assertEqual(cls, dataset.classes[dataset.class_to_idx[cls]])
        class_a_idx = dataset.class_to_idx['a']
        class_b_idx = dataset.class_to_idx['b']
        imgs_a = [(img_path, class_a_idx)for img_path in Tester.class_a_images]
        imgs_b = [(img_path, class_b_idx)for img_path in Tester.class_b_images]
        imgs = sorted(imgs_a + imgs_b)
        self.assertEqual(imgs, dataset.imgs)

        outputs = sorted([dataset[i] for i in range(len(dataset))])
        self.assertEqual(imgs, outputs)

    def test_transform(self):
        return_value = get_file_path_2('test/assets/dataset/a/a1.png')

        args = []
        transform = mock_transform(return_value, args)

        dataset = ImageFolder(Tester.root, loader=lambda x: x, transform=transform)
        outputs = [dataset[i][0] for i in range(len(dataset))]
        self.assertEqual([return_value] * len(outputs), outputs)

        imgs = sorted(Tester.class_a_images + Tester.class_b_images)
        self.assertEqual(imgs, sorted(args))

    def test_target_transform(self):
        return_value = 1

        args = []
        target_transform = mock_transform(return_value, args)

        dataset = ImageFolder(Tester.root, loader=lambda x: x, target_transform=target_transform)
        outputs = [dataset[i][1] for i in range(len(dataset))]
        self.assertEqual([return_value] * len(outputs), outputs)

        class_a_idx = dataset.class_to_idx['a']
        class_b_idx = dataset.class_to_idx['b']
        targets = sorted([class_a_idx] * len(Tester.class_a_images) +
                         [class_b_idx] * len(Tester.class_b_images))
        self.assertEqual(targets, sorted(args))


if __name__ == '__main__':
    unittest.main()
