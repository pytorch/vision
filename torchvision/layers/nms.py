from torchvision import _C


def nms(dets, scores, threshold):
    """This function performs Non-maximum suppresion"""
    return _C.nms(dets, scores, threshold)
