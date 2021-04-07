# Python examples

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/vision/blob/master/examples/python/tensor_transforms.ipynb)
[Examples of Tensor Images transformations](https://github.com/pytorch/vision/blob/master/examples/python/tensor_transforms.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/vision/blob/master/examples/python/video_api.ipynb)
[Example of VideoAPI](https://github.com/pytorch/vision/blob/master/examples/python/video_api.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/vision/blob/master/examples/python/visualization_utils.ipynb)
[Example of Visualization Utils](https://github.com/pytorch/vision/blob/master/examples/python/visualization_utils.ipynb)


Prior to v0.8.0, transforms in torchvision have traditionally been PIL-centric and presented multiple limitations due to 
that. Now, since v0.8.0, transforms implementations are Tensor and PIL compatible and we can achieve the following new 
features:
- transform multi-band torch tensor images (with more than 3-4 channels) 
- torchscript transforms together with your model for deployment
- support for GPU acceleration
- batched transformation such as for videos
- read and decode data directly as torch tensor with torchscript support (for PNG and JPEG image formats)

Furthermore, previously we used to provide a very high-level API for video decoding which left little control to the user. We're now expanding that API (and replacing it in the future) with a lower-level API that allows the user a frame-based access to a video.

Torchvision also provides utilities to visualize results. You can make grid of images, plot bounding boxes as well as segmentation masks. Thse utilities work standalone as well as with torchvision models for detection and segmentation.
