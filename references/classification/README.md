# Image classification reference training scripts

This folder contains reference training scripts for image classification.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

Except otherwise noted, all models have been trained on 8x V100 GPUs.

### ResNext-50 32x4d
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --model resnext50_32x4d --epochs 100
```


### ResNext-101 32x8d

On 8 nodes, each with 8 GPUs (for a total of 64 GPUS)
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --model resnext101_32x8d --epochs 100
```


### MobileNetV2
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
     --model mobilenet_v2 --epochs 300 --lr 0.045 --wd 0.00004\
     --lr-step-size 1 --lr-gamma 0.98
```
