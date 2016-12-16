import torch
import math

def make_grid(tensor, nrow=8, padding=2):
    """
    Given a 4D mini-batch Tensor of shape (B x C x H x W),
    or a list of images all of the same size,
    makes a grid of images
    """
    tensorlist = None
    if isinstance(tensor, list):
        tensorlist = tensor
        numImages = len(tensorlist)
        size = torch.Size(torch.Size([long(numImages)]) + tensorlist[0].size())
        tensor = tensorlist[0].new(size)
        for i in range(numImages):
            tensor[i].copy_(tensorlist[i])
    if tensor.dim() == 2: # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3: # single image
        if tensor.size(0) == 1:
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1: # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)
    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(nmaps / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps, width * xmaps).fill_(tensor.max())
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y*height+1+padding//2,height-padding)\
                .narrow(2, x*width+1+padding//2, width-padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    from PIL import Image
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding)
    ndarr = grid.mul(0.5).add(0.5).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)
