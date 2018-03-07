import unittest
from unittest.mock import Mock
import os

from torchvision.datasets import ImageFolder



class Tester(unittest.TestCase):
    root = './assets/dataset/'
    classes = ['a', 'b']
    class_a_images = [os.path.join('./assets/dataset/a/', path) for path in ['a1.png', 'a2.png', 'a3.png']]
    class_b_images = [os.path.join('./assets/dataset/b/', path) for path in ['b1.png', 'b2.png', 'b3.png', 'b4.png']]

    def test_image_folder(self):
        dataset = ImageFolder(Tester.root, loader=lambda x:x)
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
        return_value = './data/a/a1.png'
        transform = Mock(return_value=return_value)
        dataset = ImageFolder(Tester.root, loader=lambda x:x, transform=transform)
        outputs = [dataset[i][0] for i in range(len(dataset))]
        self.assertEqual([return_value]*len(outputs), outputs)

        imgs = sorted(Tester.class_a_images + Tester.class_b_images)
        args = [call[0][0] for call in transform.call_args_list]
        self.assertEqual(imgs, sorted(args))

    def test_target_transform(self):
        return_value = 1
        target_transform = Mock(return_value=return_value)
        dataset = ImageFolder(Tester.root, loader=lambda x:x, target_transform=target_transform)
        outputs = [dataset[i][1] for i in range(len(dataset))]
        self.assertEqual([return_value]*len(outputs), outputs)

        class_a_idx = dataset.class_to_idx['a']
        class_b_idx = dataset.class_to_idx['b']
        targets = sorted([class_a_idx]*len(Tester.class_a_images) +
                         [class_b_idx]*len(Tester.class_b_images))
        args = [call[0][0] for call in target_transform.call_args_list]
        self.assertEqual(targets, sorted(args))

if __name__ == '__main__':
    unittest.main()
