from .lsun import LSUNDataset, LSUNClassDataset
from .folder import ImageFolderDataset
from .coco import CocoCaptionsDataset, CocoDetectionDataset

__all__ = ('LSUNDataset', 'LSUNClassDataset',
           'ImageFolderDataset',
           'CocoCaptionsDataset', 'CocoDetectionDataset')
