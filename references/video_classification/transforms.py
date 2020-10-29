def bhwc_to_bchw(vid):
    """Convert tensor from (B, H, W, C) to (B, C, H, W)
    """
    return vid.permute(0, 3, 1, 2)


def bchw_to_cbhw(vid):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)
    """
    return vid.permute(1, 0, 2, 3)
