import cvtransforms
import transforms
import unittest
import numpy as np
import torch
import cv2

class TestOpenCVTransforms(unittest.TestCase):
    def testScale(self):
        size = 43
        w, h = 68, 54
        img = np.random.randn(h, w, 3)
        tr = cvtransforms.Scale(size)
        res = tr(img)
        self.assertEqual(res.shape[0], size)
        self.assertEqual(res.shape[1], h)

    def testCenterCrop(self):
        size = 43
        w, h = 68, 54
        img = np.random.randn(h, w, 3)
        tr = cvtransforms.CenterCrop(size)
        res = tr(img)
        self.assertEqual(res.shape[0], size)
        self.assertEqual(res.shape[1], size)

    def testNormalize(self):
        meanstd = dict(mean=[1,2,3], std=[1,1,1])
        normalize = transforms.Normalize(**meanstd)
        cvnormalize = cvtransforms.Normalize(**meanstd)
                                         
        w, h = 68, 54
        img = np.random.randn(h, w, 3)
        for i in range(3):
            img[:,:,i] = i+1
        res_th = normalize(torch.from_numpy(img).clone().permute(2,0,1)).permute(1,2,0).numpy()
        res_np = cvnormalize(img)
        self.assertEqual(np.abs(res_np - res_th).sum(), 0)

    def testFlip(self):
        w, h = 12, 10
        img = np.random.randn(h, w, 1)
        img[:,:6,:] = 0
        img[:,6:,:] = 1

        flip = img
        while id(flip) == id(img):
            flip = cvtransforms.RandomHorizontalFlip()(img)
        self.assertEqual(flip[:,:6,:].mean(), 1)
        self.assertEqual(flip[:,6:,:].mean(), 0)

    def testPadding(self):
        w, h = 12, 10
        img = np.random.randn(h, w, 1)
        img[:,:6,:] = 0
        img[:,6:,:] = 1

        padded = cvtransforms.Pad(2, cv2.BORDER_REFLECT)(img)
        self.assertEqual(padded[:,:8,:].mean(), 0)
        self.assertEqual(padded[:,8:,:].mean(), 1)


if __name__ == '__main__':
    unittest.main()
