import unittest

import torch
import torchvision.transforms.functional_tensor as F_t


class Tester(unittest.TestCase):
	def setUp(self):
		self.device = "cpu"

	def test_get_image_size(self):
		scripted_fn = torch.jit.script(F_t._get_image_size)

		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor)
		self.assertTrue('Tensor is not a torch image.' in str(context.exception))


		shape = (100, 100)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		self.assertTrue([shape[-1], shape[-2]], scripted_fn(tensor))

	def test_vflip(self):
		scripted_fn = torch.jit.script(F_t.vflip)
		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor)
		self.assertTrue('Tensor is not a torch image.' in str(context.exception))

	def test_hflip(self):
		scripted_fn = torch.jit.script(F_t.hflip)
		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor)
		self.assertTrue('Tensor is not a torch image.' in str(context.exception))

	def test_crop(self):
		scripted_fn = torch.jit.script(F_t.crop)
		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor, 1, 2, 4, 5)
		self.assertTrue('Tensor is not a torch image.' in str(context.exception))

	def test_adjust_brightness(self):
		scripted_fn = torch.jit.script(F_t.adjust_brightness)
		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor, 0.)
		self.assertTrue('Tensor is not a torch image.' in str(context.exception))

	def test_adjust_contrast(self):
		scripted_fn = torch.jit.script(F_t.adjust_contrast)
		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor, 1.)
		self.assertTrue('Tensor is not a torch image.' in str(context.exception))

	def test_adjust_hue(self):
		scripted_fn = torch.jit.script(F_t.adjust_hue)
		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor, -0.5)
		self.assertTrue('Tensor is not a torch image.' in str(context.exception))


	def test_adjust_saturation(self):
		scripted_fn = torch.jit.script(F_t.adjust_saturation)
		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor, 2.)
		self.assertTrue('Tensor is not a torch image.' in str(context.exception))

	def test_center_crop(self):
		scripted_fn = torch.jit.script(F_t.center_crop)
		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor, [10, 11])
		self.assertTrue('Tensor is not a torch image.' in str(context.exception))

	def test_five_crop(self):
		scripted_fn = torch.jit.script(F_t.five_crop)
		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor, [10, 11])
		self.assertTrue('Tensor is not a torch image.' in str(context.exception))

	def test_ten_crop(self):
		scripted_fn = torch.jit.script(F_t.ten_crop)
		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor, [10, 11])
		self.assertTrue('Tensor is not a torch image.' in str(context.exception))

	#need to fix
	def test_pad(self):
		scripted_fn = torch.jit.script(F_t.pad)
		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		# config = {"padding_mode": "constant", "fill": 0}
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor, [2, ],  2, "constant")

		self.assertTrue('Tensor is not a torch image.' in str(context.exception))

	def test_resize(self):
		scripted_fn = torch.jit.script(F_t.resize)
		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor, [10, 11])
		self.assertTrue('Tensor is not a torch image.' in str(context.exception))


	def test_perspective(self):
		scripted_fn = torch.jit.script(F_t.perspective)
		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor, [0.2, ])
		self.assertTrue('Tensor is not a torch image.' in str(context.exception))


	def test_gaussian_blur(self):
		scripted_fn = torch.jit.script(F_t.gaussian_blur)
		shape = (100,)
		tensor = torch.rand(*shape, dtype=torch.float, device=self.device)
		with self.assertRaises(Exception) as context:
			scripted_fn(tensor, (2, 2), (0.7, 0.5))
		self.assertTrue('Tensor is not a torch image.' in str(context.exception))




@unittest.skipIf(not torch.cuda.is_available(), reason="Skip if no CUDA device")
class CUDATester(Tester):

	def setUp(self):
		self.device = "cuda"



if __name__ == '__main__':
	unittest.main()