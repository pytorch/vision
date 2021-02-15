from torch import nn, Tensor

from typing import Dict, List, Optional, Tuple

__all__ = ['SSD']


class SSDHead(nn.Module):
    # TODO: Similar to RetinaNetHead. Perhaps abstract and reuse for one-shot detectors.
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor],
                     matched_idxs: List[Tensor]) -> Dict[str, Tensor]:
        return {
            'classification': self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            'bbox_regression': self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            'cls_logits': self.classification_head(x),
            'bbox_regression': self.regression_head(x)
        }


class SSDClassificationHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor],
                     matched_idxs: List[Tensor]) -> Tensor:
        pass

    def forward(self, x: List[Tensor]) -> Tensor:
        pass


class SSDRegressionHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor],
                     matched_idxs: List[Tensor]) -> Tensor:
        pass

    def forward(self, x: List[Tensor]) -> Tensor:
        pass


class SSD(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor],
                     anchors: List[Tensor]) -> Dict[str, Tensor]:
        pass

    def postprocess_detections(self, head_outputs: Dict[str, List[Tensor]], anchors: List[List[Tensor]],
                               image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]:
        pass

    def forward(self, images: List[Tensor],
                targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        pass
