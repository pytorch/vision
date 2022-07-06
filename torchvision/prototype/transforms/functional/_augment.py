from torchvision.transforms import functional_tensor as _FT


erase_image_tensor = _FT.erase


# TODO: Don't forget to clean up from the primitives kernels those that shouldn't be kernels.
# Like the mixup and cutmix stuff

# This function is copy-pasted to Image and OneHotLabel and may be refactored
# def _mixup_tensor(input: torch.Tensor, batch_dim: int, lam: float) -> torch.Tensor:
#     input = input.clone()
#     return input.roll(1, batch_dim).mul_(1 - lam).add_(input.mul_(lam))
