import warnings


def _raise_video_deprecation_warning():
    warnings.warn(
        "The video decoding and encoding capabilities of torchvision "
        "are deprecated from version 0.22 and will be removed in version 0.24. "
        "We recommend that you migrate to TorchCodec, where we'll consolidate "
        "the future decoding/encoding capabilities of PyTorch: "
        "https://github.com/pytorch/torchcodec",
        UserWarning,
    )
