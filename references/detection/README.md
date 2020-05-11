# Object detection reference training scripts

This folder contains reference training scripts for object detection.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

Except otherwise noted, all models have been trained on 8x V100 GPUs.

### Faster R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```


### Mask R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```


### Keypoint R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 46\
    --lr-steps 36 43 --aspect-ratio-group-factor 3
```

# External repositories

For further examples and explanations on how to use the object detection code in this folder, see Microsoft's [Computer Vision Best Practices](https://github.com/microsoft/computervision-recipes/tree/master/scenarios/detection) repository. The repository contains multiple notebooks with functionality to help with e.g. training on a custom dataset, model evaluation, parameters selection, etc., to more advanced topics such as hard-negative mining or model deployment.

<p align="center">
  <img src="https://cvbp.blob.core.windows.net/public/images/cvbp_notebooks.jpg" width="700" alt="CVBP notebooks"/>
</p>
