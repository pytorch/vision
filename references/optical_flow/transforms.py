import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


class ValidateModelInput(torch.nn.Module):
    # Pass-through transform that checks the shape and dtypes to make sure the model gets what it expects
    def forward(self, img1, img2, flow, valid_flow_mask):

        if not all(isinstance(arg, torch.Tensor) for arg in (img1, img2, flow, valid_flow_mask) if arg is not None):
            raise TypeError("This method expects all input arguments to be of type torch.Tensor.")
        if not all(arg.dtype == torch.float32 for arg in (img1, img2, flow) if arg is not None):
            raise TypeError("This method expects the tensors img1, img2 and flow of be of dtype torch.float32.")

        if img1.shape != img2.shape:
            raise ValueError("img1 and img2 should have the same shape.")
        h, w = img1.shape[-2:]
        if flow is not None and flow.shape != (2, h, w):
            raise ValueError(f"flow.shape should be (2, {h}, {w}) instead of {flow.shape}")
        if valid_flow_mask is not None:
            if valid_flow_mask.shape != (h, w):
                raise ValueError(f"valid_flow_mask.shape should be ({h}, {w}) instead of {valid_flow_mask.shape}")
            if valid_flow_mask.dtype != torch.bool:
                raise TypeError("valid_flow_mask should be of dtype torch.bool instead of {valid_flow_mask.dtype}")

        return img1, img2, flow, valid_flow_mask


class MakeValidFlowMask(torch.nn.Module):
    # This transform generates a valid_flow_mask if it doesn't exist.
    # The flow is considered valid if ||flow||_inf < threshold
    # This is a noop for Kitti and HD1K which already come with a built-in flow mask.
    def __init__(self, threshold=1000):
        super().__init__()
        self.threshold = threshold

    def forward(self, img1, img2, flow, valid_flow_mask):
        if flow is not None and valid_flow_mask is None:
            valid_flow_mask = (flow.abs() < self.threshold).all(axis=0)
        return img1, img2, flow, valid_flow_mask


class ConvertImageDtype(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, img1, img2, flow, valid_flow_mask):
        img1 = F.convert_image_dtype(img1, dtype=self.dtype)
        img2 = F.convert_image_dtype(img2, dtype=self.dtype)

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        return img1, img2, flow, valid_flow_mask


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img1, img2, flow, valid_flow_mask):
        img1 = F.normalize(img1, mean=self.mean, std=self.std)
        img2 = F.normalize(img2, mean=self.mean, std=self.std)

        return img1, img2, flow, valid_flow_mask


class PILToTensor(torch.nn.Module):
    # Converts all inputs to tensors
    # Technically the flow and the valid mask are numpy arrays, not PIL images, but we keep that naming
    # for consistency with the rest, e.g. the segmentation reference.
    def forward(self, img1, img2, flow, valid_flow_mask):
        img1 = F.pil_to_tensor(img1)
        img2 = F.pil_to_tensor(img2)
        if flow is not None:
            flow = torch.from_numpy(flow)
        if valid_flow_mask is not None:
            valid_flow_mask = torch.from_numpy(valid_flow_mask)

        return img1, img2, flow, valid_flow_mask


class AsymmetricColorJitter(T.ColorJitter):
    # p determines the proba of doing asymmertric vs symmetric color jittering
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.2):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.p = p

    def forward(self, img1, img2, flow, valid_flow_mask):

        if torch.rand(1) < self.p:
            # asymmetric: different transform for img1 and img2
            img1 = super().forward(img1)
            img2 = super().forward(img2)
        else:
            # symmetric: same transform for img1 and img2
            batch = torch.stack([img1, img2])
            batch = super().forward(batch)
            img1, img2 = batch[0], batch[1]

        return img1, img2, flow, valid_flow_mask


class RandomErasing(T.RandomErasing):
    # This only erases img2, and with an extra max_erase param
    # This max_erase is needed because in the RAFT training ref does:
    # 0 erasing with .5 proba
    # 1 erase with .25 proba
    # 2 erase with .25 proba
    # and there's no accurate way to achieve this otherwise.
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False, max_erase=1):
        super().__init__(p=p, scale=scale, ratio=ratio, value=value, inplace=inplace)
        self.max_erase = max_erase
        if self.max_erase <= 0:
            raise ValueError("max_raise should be greater than 0")

    def forward(self, img1, img2, flow, valid_flow_mask):
        if torch.rand(1) > self.p:
            return img1, img2, flow, valid_flow_mask

        for _ in range(torch.randint(self.max_erase, size=(1,)).item()):
            x, y, h, w, v = self.get_params(img2, scale=self.scale, ratio=self.ratio, value=[self.value])
            img2 = F.erase(img2, x, y, h, w, v, self.inplace)

        return img1, img2, flow, valid_flow_mask


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, img1, img2, flow, valid_flow_mask):
        if torch.rand(1) > self.p:
            return img1, img2, flow, valid_flow_mask

        img1 = F.hflip(img1)
        img2 = F.hflip(img2)
        flow = F.hflip(flow) * torch.tensor([-1, 1])[:, None, None]
        if valid_flow_mask is not None:
            valid_flow_mask = F.hflip(valid_flow_mask)
        return img1, img2, flow, valid_flow_mask


class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(self, img1, img2, flow, valid_flow_mask):
        if torch.rand(1) > self.p:
            return img1, img2, flow, valid_flow_mask

        img1 = F.vflip(img1)
        img2 = F.vflip(img2)
        flow = F.vflip(flow) * torch.tensor([1, -1])[:, None, None]
        if valid_flow_mask is not None:
            valid_flow_mask = F.vflip(valid_flow_mask)
        return img1, img2, flow, valid_flow_mask


class RandomResizeAndCrop(torch.nn.Module):
    # This transform will resize the input with a given proba, and then crop it.
    # These are the reversed operations of the built-in RandomResizedCrop,
    # although the order of the operations doesn't matter too much: resizing a
    # crop would give the same result as cropping a resized image, up to
    # interpolation artifact at the borders of the output.
    #
    # The reason we don't rely on RandomResizedCrop is because of a significant
    # difference in the parametrization of both transforms, in particular,
    # because of the way the random parameters are sampled in both transforms,
    # which leads to fairly different results (and different epe). For more details see
    # https://github.com/pytorch/vision/pull/5026/files#r762932579
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, stretch_prob=0.8):
        super().__init__()
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.stretch_prob = stretch_prob
        self.resize_prob = 0.8
        self.max_stretch = 0.2

    def forward(self, img1, img2, flow, valid_flow_mask):
        # randomly sample scale
        h, w = img1.shape[-2:]
        # Note: in original code, they use + 1 instead of + 8 for sparse datasets (e.g. Kitti)
        # It shouldn't matter much
        min_scale = max((self.crop_size[0] + 8) / h, (self.crop_size[1] + 8) / w)

        scale = 2 ** torch.empty(1, dtype=torch.float32).uniform_(self.min_scale, self.max_scale).item()
        scale_x = scale
        scale_y = scale
        if torch.rand(1) < self.stretch_prob:
            scale_x *= 2 ** torch.empty(1, dtype=torch.float32).uniform_(-self.max_stretch, self.max_stretch).item()
            scale_y *= 2 ** torch.empty(1, dtype=torch.float32).uniform_(-self.max_stretch, self.max_stretch).item()

        scale_x = max(scale_x, min_scale)
        scale_y = max(scale_y, min_scale)

        new_h, new_w = round(h * scale_y), round(w * scale_x)

        if torch.rand(1).item() < self.resize_prob:
            # rescale the images
            # We hard-code antialias=False to preserve results after we changed
            # its default from None to True (see
            # https://github.com/pytorch/vision/pull/7160)
            # TODO: we could re-train the OF models with antialias=True?
            img1 = F.resize(img1, size=(new_h, new_w), antialias=False)
            img2 = F.resize(img2, size=(new_h, new_w), antialias=False)
            if valid_flow_mask is None:
                flow = F.resize(flow, size=(new_h, new_w))
                flow = flow * torch.tensor([scale_x, scale_y])[:, None, None]
            else:
                flow, valid_flow_mask = self._resize_sparse_flow(
                    flow, valid_flow_mask, scale_x=scale_x, scale_y=scale_y
                )

        # Note: For sparse datasets (Kitti), the original code uses a "margin"
        # See e.g. https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py#L220:L220
        # We don't, not sure if it matters much
        y0 = torch.randint(0, img1.shape[1] - self.crop_size[0], size=(1,)).item()
        x0 = torch.randint(0, img1.shape[2] - self.crop_size[1], size=(1,)).item()

        img1 = F.crop(img1, y0, x0, self.crop_size[0], self.crop_size[1])
        img2 = F.crop(img2, y0, x0, self.crop_size[0], self.crop_size[1])
        flow = F.crop(flow, y0, x0, self.crop_size[0], self.crop_size[1])
        if valid_flow_mask is not None:
            valid_flow_mask = F.crop(valid_flow_mask, y0, x0, self.crop_size[0], self.crop_size[1])

        return img1, img2, flow, valid_flow_mask

    def _resize_sparse_flow(self, flow, valid_flow_mask, scale_x=1.0, scale_y=1.0):
        # This resizes both the flow and the valid_flow_mask mask (which is assumed to be reasonably sparse)
        # There are as-many non-zero values in the original flow as in the resized flow (up to OOB)
        # So for example if scale_x = scale_y = 2, the sparsity of the output flow is multiplied by 4

        h, w = flow.shape[-2:]

        h_new = int(round(h * scale_y))
        w_new = int(round(w * scale_x))
        flow_new = torch.zeros(size=[2, h_new, w_new], dtype=flow.dtype)
        valid_new = torch.zeros(size=[h_new, w_new], dtype=valid_flow_mask.dtype)

        jj, ii = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")

        ii_valid, jj_valid = ii[valid_flow_mask], jj[valid_flow_mask]

        ii_valid_new = torch.round(ii_valid.to(float) * scale_y).to(torch.long)
        jj_valid_new = torch.round(jj_valid.to(float) * scale_x).to(torch.long)

        within_bounds_mask = (0 <= ii_valid_new) & (ii_valid_new < h_new) & (0 <= jj_valid_new) & (jj_valid_new < w_new)

        ii_valid = ii_valid[within_bounds_mask]
        jj_valid = jj_valid[within_bounds_mask]
        ii_valid_new = ii_valid_new[within_bounds_mask]
        jj_valid_new = jj_valid_new[within_bounds_mask]

        valid_flow_new = flow[:, ii_valid, jj_valid]
        valid_flow_new[0] *= scale_x
        valid_flow_new[1] *= scale_y

        flow_new[:, ii_valid_new, jj_valid_new] = valid_flow_new
        valid_new[ii_valid_new, jj_valid_new] = 1

        return flow_new, valid_new


class Compose(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, img1, img2, flow, valid_flow_mask):
        for t in self.transforms:
            img1, img2, flow, valid_flow_mask = t(img1, img2, flow, valid_flow_mask)
        return img1, img2, flow, valid_flow_mask
