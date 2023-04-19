import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from ...ops import batched_nms
from ...transforms import functional as F
from .._api import register_model, Weights, WeightsEnum
from .._utils import _ovewrite_value_param
from ..yolo import YOLOV4Backbone
from .backbone_utils import _validate_trainable_layers
from .yolo_networks import DarknetNetwork, PRED, TARGET, TARGETS, YOLOV4Network

IMAGES = List[Tensor]  # TorchScript doesn't allow a tuple.


class YOLO(nn.Module):
    """YOLO implementation that supports the most important features of YOLOv3, YOLOv4, YOLOv5, YOLOv7, Scaled-
    YOLOv4, and YOLOX.

    *YOLOv3 paper*: `Joseph Redmon and Ali Farhadi <https://arxiv.org/abs/1804.02767>`__

    *YOLOv4 paper*: `Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao <https://arxiv.org/abs/2004.10934>`__

    *YOLOv7 paper*: `Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao <https://arxiv.org/abs/2207.02696>`__

    *Scaled-YOLOv4 paper*: `Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao
    <https://arxiv.org/abs/2011.08036>`__

    *YOLOX paper*: `Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun <https://arxiv.org/abs/2107.08430>`__

    The network architecture can be written in PyTorch, or read from a Darknet configuration file using the
    :class:`~.yolo_networks.DarknetNetwork` class. ``DarknetNetwork`` is also able to read weights that have been saved
    by Darknet.

    The input is expected to be a list of images. Each image is a tensor with shape ``[channels, height, width]``. The
    images from a single batch will be stacked into a single tensor, so the sizes have to match. Different batches can
    have different image sizes, as long as the size is divisible by the ratio in which the network downsamples the
    input.

    During training, the model expects both the image tensors and a list of targets. It's possible to train a model
    using one integer class label per box, but the YOLO model supports also multiple labels per box. For multi-label
    training, simply use a boolean matrix that indicates which classes are assigned to which boxes, in place of the
    class labels. *Each target is a dictionary containing the following tensors*:

    - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in `(x1, y1, x2, y2)` format
    - labels (``Int64Tensor[N]`` or ``BoolTensor[N, classes]``): the class label or a boolean class mask for each
      ground-truth box

    :func:`~.yolo.YOLO.forward` method returns all predictions from all detection layers in one tensor with shape
    ``[N, anchors, classes + 5]``, where ``anchors`` is the total number of anchors in all detection layers. The
    coordinates are scaled to the input image size. During training it also returns a dictionary containing the
    classification, box overlap, and confidence losses.

    During inference, the model requires only the image tensor. :func:`~.yolo.YOLO.infer` method filters and
    processes the predictions. If a prediction has a high score for more than one class, it will be duplicated. *The
    processed output is returned in a dictionary containing the following tensors*:

    - boxes (``FloatTensor[N, 4]``): predicted bounding box `(x1, y1, x2, y2)` coordinates in image space
    - scores (``FloatTensor[N]``): detection confidences
    - labels (``Int64Tensor[N]``): the predicted labels for each object

    Detection using a Darknet configuration and pretrained weights:

        >>> from urllib.request import urlretrieve
        >>> import torch
        >>> from torchvision.models.detection import DarknetNetwork, YOLO
        >>>
        >>> urlretrieve("https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny-3l.cfg", "yolov4-tiny-3l.cfg")
        >>> urlretrieve("https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29", "yolov4-tiny.conv.29")
        >>> network = DarknetNetwork("yolov4-tiny-3l.cfg", "yolov4-tiny.conv.29")
        >>> model = YOLO(network)
        >>> image = torch.rand(3, 608, 608)
        >>> detections = model.infer(image)

    Detection using a predefined YOLOv4 network:

        >>> import torch
        >>> from torchvision.models.detection import YOLOV4Network, YOLO
        >>>
        >>> network = YOLOV4Network(num_classes=91)
        >>> model = YOLO(network)
        >>> image = torch.rand(3, 608, 608)
        >>> detections = model.infer(image)

    Args:
        network: A module that represents the network layers. This can be obtained from a Darknet configuration using
            :func:`~.yolo_networks.DarknetNetwork`, or it can be defined as PyTorch code.
        confidence_threshold: Postprocessing will remove bounding boxes whose confidence score is not higher than this
            threshold.
        nms_threshold: Non-maximum suppression will remove bounding boxes whose IoU with a higher confidence box is
            higher than this threshold, if the predicted categories are equal.
        detections_per_image: Keep at most this number of highest-confidence detections per image.
    """

    def __init__(
        self,
        network: nn.Module,
        confidence_threshold: float = 0.2,
        nms_threshold: float = 0.45,
        detections_per_image: int = 300,
    ) -> None:
        super().__init__()

        self.network = network
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.detections_per_image = detections_per_image

    def forward(
        self, images: Union[Tensor, IMAGES], targets: Optional[TARGETS] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor, List[int]]]:
        """Runs a forward pass through the network (all layers listed in ``self.network``), and if training targets
        are provided, computes the losses from the detection layers.

        Detections are concatenated from the detection layers. Each detection layer will produce a number of detections
        that depends on the size of the feature map and the number of anchors per feature map cell.

        Args:
            images: A tensor of size ``[batch_size, channels, height, width]`` containing a batch of images or a list of
                image tensors.
            targets: If given, computes losses from detection layers against these targets. A list of target
                dictionaries, one for each image.

        Returns:
            detections (:class:`~torch.Tensor`), losses (:class:`~torch.Tensor`), hits (List[int]): Detections, and if
            targets were provided, a dictionary of losses. Detections are shaped ``[batch_size, anchors, classes + 5]``,
            where ``anchors`` is the feature map size (width * height) times the number of anchors per cell. The
            predicted box coordinates are in `(x1, y1, x2, y2)` format and scaled to the input image size.
        """
        self.validate_batch(images, targets)
        images_tensor = images if isinstance(images, Tensor) else torch.stack(images)
        detections, losses, hits = self.network(images_tensor, targets)

        detections = torch.cat(detections, 1)
        if targets is None:
            return detections

        losses = torch.stack(losses).sum(0)
        return detections, losses, hits

    def infer(self, image: Tensor) -> PRED:
        """Feeds an image to the network and returns the detected bounding boxes, confidence scores, and class
        labels.

        If a prediction has a high score for more than one class, it will be duplicated.

        Args:
            image: An input image, a tensor of uint8 values sized ``[channels, height, width]``.

        Returns:
            A dictionary containing tensors "boxes", "scores", and "labels". "boxes" is a matrix of detected bounding
            box `(x1, y1, x2, y2)` coordinates. "scores" is a vector of confidence scores for the bounding box
            detections. "labels" is a vector of predicted class labels.
        """
        if not isinstance(image, Tensor):
            image = F.to_tensor(image)

        was_training = self.training
        self.eval()

        detections = self([image])
        detections = self.process_detections(detections)
        detections = detections[0]

        if was_training:
            self.train()
        return detections

    def process_detections(self, preds: Tensor) -> List[PRED]:
        """Splits the detection tensor returned by a forward pass into a list of prediction dictionaries, and
        filters them based on confidence threshold, non-maximum suppression (NMS), and maximum number of
        predictions.

        If for any single detection there are multiple categories whose score is above the confidence threshold, the
        detection will be duplicated to create one detection for each category. NMS processes one category at a time,
        iterating over the bounding boxes in descending order of confidence score, and removes lower scoring boxes that
        have an IoU greater than the NMS threshold with a higher scoring box.

        The returned detections are sorted by descending confidence. The items of the dictionaries are as follows:
        - boxes (``Tensor[batch_size, N, 4]``): detected bounding box `(x1, y1, x2, y2)` coordinates
        - scores (``Tensor[batch_size, N]``): detection confidences
        - labels (``Int64Tensor[batch_size, N]``): the predicted class IDs

        Args:
            preds: A tensor of detected bounding boxes and their attributes.

        Returns:
            Filtered detections. A list of prediction dictionaries, one for each image.
        """

        def process(boxes: Tensor, confidences: Tensor, classprobs: Tensor) -> Dict[str, Any]:
            scores = classprobs * confidences[:, None]

            # Select predictions with high scores. If a prediction has a high score for more than one class, it will be
            # duplicated.
            idxs, labels = (scores > self.confidence_threshold).nonzero().T
            boxes = boxes[idxs]
            scores = scores[idxs, labels]

            keep = batched_nms(boxes, scores, labels, self.nms_threshold)
            keep = keep[: self.detections_per_image]
            return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}

        return [process(p[..., :4], p[..., 4], p[..., 5:]) for p in preds]

    def process_targets(self, targets: TARGETS) -> List[TARGET]:
        """Duplicates multi-label targets to create one target for each label.

        Args:
            targets: List of target dictionaries. Each dictionary must contain "boxes" and "labels". "labels" is either
                a one-dimensional list of class IDs, or a two-dimensional boolean class map.

        Returns:
            Single-label targets. A list of target dictionaries, one for each image.
        """

        def process(boxes: Tensor, labels: Tensor, **other: Any) -> Dict[str, Any]:
            if labels.ndim == 2:
                idxs, labels = labels.nonzero().T
                boxes = boxes[idxs]
            return {"boxes": boxes, "labels": labels, **other}

        return [process(**t) for t in targets]

    def validate_batch(self, images: Union[Tensor, IMAGES], targets: Optional[TARGETS]) -> None:
        """Validates the format of a batch of data.

        Args:
            images: A tensor containing a batch of images or a list of image tensors.
            targets: A list of target dictionaries or ``None``. If a list is provided, there should be as many target
                dictionaries as there are images.
        """
        if not isinstance(images, Tensor):
            if not isinstance(images, (tuple, list)):
                raise TypeError(f"Expected images to be a Tensor, tuple, or a list, got {type(images).__name__}.")
            if not images:
                raise ValueError("No images in batch.")
            shape = images[0].shape
            for image in images:
                if not isinstance(image, Tensor):
                    raise ValueError(f"Expected image to be of type Tensor, got {type(image).__name__}.")
                if image.shape != shape:
                    raise ValueError(f"Images with different shapes in one batch: {shape} and {image.shape}")

        if targets is None:
            if self.training:
                raise ValueError("Targets should be given in training mode.")
            else:
                return

        if not isinstance(targets, (tuple, list)):
            raise TypeError(f"Expected targets to be a tuple or a list, got {type(images).__name__}.")
        if len(images) != len(targets):
            raise ValueError(f"Got {len(images)} images, but targets for {len(targets)} images.")

        for target in targets:
            if "boxes" not in target:
                raise ValueError("Target dictionary doesn't contain boxes.")
            boxes = target["boxes"]
            if not isinstance(boxes, Tensor):
                raise TypeError(f"Expected target boxes to be of type Tensor, got {type(boxes).__name__}.")
            if (boxes.ndim != 2) or (boxes.shape[-1] != 4):
                raise ValueError(f"Expected target boxes to be tensors of shape [N, 4], got {list(boxes.shape)}.")
            if "labels" not in target:
                raise ValueError("Target dictionary doesn't contain labels.")
            labels = target["labels"]
            if not isinstance(labels, Tensor):
                raise ValueError(f"Expected target labels to be of type Tensor, got {type(labels).__name__}.")
            if (labels.ndim < 1) or (labels.ndim > 2) or (len(labels) != len(boxes)):
                raise ValueError(
                    f"Expected target labels to be tensors of shape [N] or [N, num_classes], got {list(labels.shape)}."
                )


class YOLOV4_Backbone_Weights(WeightsEnum):
    # TODO: Create pretrained weights.
    DEFAULT = Weights(
        url="",
        transforms=lambda x: x,
        meta={},
    )


class YOLOV4_Weights(WeightsEnum):
    # TODO: Create pretrained weights.
    DEFAULT = Weights(
        url="",
        transforms=lambda x: x,
        meta={},
    )


def freeze_backbone_layers(backbone: nn.Module, trainable_layers: Optional[int], is_trained: bool) -> None:
    """Freezes backbone layers layers that won't be used for training.

    Args:
        backbone: The backbone network.
        trainable_layers: Number of trainable layers (stages), starting from the final stage.
        is_trained: Set to ``True`` when using pre-trained weights. Otherwise will issue a warning if
            ``trainable_layers`` is set.
    """
    if not hasattr(backbone, "stages"):
        warnings.warn("Cannot freeze backbone layers. Backbone object has no 'stages' attribute.")
    num_layers = len(backbone.stages)  # type: ignore
    trainable_layers = _validate_trainable_layers(is_trained, trainable_layers, num_layers, 3)

    layers_to_train = [f"stages.{idx}" for idx in range(num_layers - trainable_layers, num_layers)]
    if trainable_layers == num_layers:
        layers_to_train.append("stem")

    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)


@register_model()
def yolov4(
    weights: Optional[YOLOV4_Weights] = None,
    progress: bool = True,
    in_channels: int = 3,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[YOLOV4_Backbone_Weights] = None,
    trainable_backbone_layers: Optional[int] = None,
    confidence_threshold: float = 0.2,
    nms_threshold: float = 0.45,
    detections_per_image: int = 300,
    **kwargs: Any,
) -> YOLO:
    """
    Constructs a YOLOv4 model.

    .. betastatus:: detection module

    Example:

        >>> import torch
        >>> from torchvision.models.detection import yolov4, YOLOV4_Weights
        >>>
        >>> model = yolov4(weights=YOLOV4_Weights.DEFAULT)
        >>> image = torch.rand(3, 608, 608)
        >>> detections = model.infer(image)

    Args:
        weights: Pretrained weights to use. See :class:`~.YOLOV4_Weights` below for more details and possible values. By
            default, the model will be initialized randomly.
        progress: If ``True``, displays a progress bar of the download to ``stderr``.
        in_channels: Number of channels in the input image.
        num_classes: Number of output classes of the model (including the background). By default, this value is set to
            91 or read from the weights.
        weights_backbone: Pretrained weights for the backbone. See :class:`~.YOLOV4_Backbone_Weights` below for more
            details and possible values. By default, the backbone will be initialized randomly.
        trainable_backbone_layers: Number of trainable (not frozen) layers (stages), starting from the final stage.
            Valid values are between 0 and the number of stages in the backbone. By default, this value is set to 3.
        confidence_threshold: Postprocessing will remove bounding boxes whose confidence score is not higher than this
            threshold.
        nms_threshold: Non-maximum suppression will remove bounding boxes whose IoU with a higher confidence box is
            higher than this threshold, if the predicted categories are equal.
        detections_per_image: Keep at most this number of highest-confidence detections per image.
        **kwargs: Parameters passed to the ``torchvision.models.detection.YOLOV4Network`` class. Please refer to the
            `source code <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/yolo_networks.py>`_
            for more details about this class.

    .. autoclass:: .YOLOV4_Weights
        :members:

    .. autoclass:: .YOLOV4_Backbone_Weights
        :members:
    """
    weights = YOLOV4_Weights.verify(weights)
    weights_backbone = YOLOV4_Backbone_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    backbone_kwargs = {key: kwargs[key] for key in ("widths", "activation", "normalization") if key in kwargs}
    backbone = YOLOV4Backbone(in_channels, **backbone_kwargs)

    is_trained = weights is not None or weights_backbone is not None
    freeze_backbone_layers(backbone, trainable_backbone_layers, is_trained)

    if weights_backbone is not None:
        backbone.load_state_dict(weights_backbone.get_state_dict(progress=progress))

    network = YOLOV4Network(num_classes, backbone, **kwargs)
    model = YOLO(network, confidence_threshold, nms_threshold, detections_per_image)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def yolo_darknet(
    config_path: str,
    weights_path: Optional[str] = None,
    confidence_threshold: float = 0.2,
    nms_threshold: float = 0.45,
    detections_per_image: int = 300,
    **kwargs: Any,
) -> YOLO:
    """
    Constructs a YOLO model from a Darknet configuration file.

    .. betastatus:: detection module

    Example:

        >>> from urllib.request import urlretrieve
        >>> from torchvision.models.detection import yolo_darknet
        >>>
        >>> urlretrieve("https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny-3l.cfg", "yolov4-tiny-3l.cfg")
        >>> urlretrieve("https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29", "yolov4-tiny.conv.29")
        >>> model = yolo_darknet("yolov4-tiny-3l.cfg", "yolov4-tiny.conv.29")
        >>> image = torch.rand(3, 608, 608)
        >>> detections = model.infer(image)

    Args:
        config_path: Path to a Darknet configuration file that defines the network architecture.
        weights_path: Path to a Darknet weights file to load.
        confidence_threshold: Postprocessing will remove bounding boxes whose confidence score is not higher than this
            threshold.
        nms_threshold: Non-maximum suppression will remove bounding boxes whose IoU with a higher confidence box is
            higher than this threshold, if the predicted categories are equal.
        detections_per_image: Keep at most this number of highest-confidence detections per image.
        **kwargs: Parameters passed to the ``torchvision.models.detection.DarknetNetwork`` class. Please refer to the
            `source code <https://github.com/pytorch/vision/blob/main/torchvision/models/detection/yolo_networks.py>`_
            for more details about this class.
    """
    network = DarknetNetwork(config_path, weights_path, **kwargs)
    return YOLO(network, confidence_threshold, nms_threshold, detections_per_image)
