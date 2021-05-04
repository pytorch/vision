# Object detection reference training scripts

This folder contains reference training scripts for object detection.
They serve as a log of how to train specific models, to provide baseline
training and evaluation scripts to quickly bootstrap research.

To execute the example commands below you must install the following:

```
cython
pycocotools
matplotlib
```

You must modify the following flags:

`--data-path=/path/to/coco/dataset`

`--nproc_per_node=<number_of_gpus_available>`

Except otherwise noted, all models have been trained on 8x V100 GPUs. 

### Faster R-CNN ResNet-50 FPN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

### Faster R-CNN MobileNetV3-Large FPN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_mobilenet_v3_large_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

### Faster R-CNN MobileNetV3-Large 320 FPN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_mobilenet_v3_large_320_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

### RetinaNet
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model retinanet_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01
```

### SSD VGG16
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model ssd300_vgg16 --epochs 120\
    --lr-steps 80 110 --aspect-ratio-group-factor 3 --lr 0.002 --batch-size 4\
    --weight-decay 0.0005 --data-augmentation ssd
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

