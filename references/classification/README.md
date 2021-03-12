# Image classification reference training scripts

This folder contains reference training scripts for image classification.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

Except otherwise noted, all models have been trained on 8x V100 GPUs with 
the following parameters:

| Parameter                | value  |
| ------------------------ | ------ |
| `--batch_size`           | `32`   |
| `--epochs`               | `90`   |
| `--lr`                   | `0.1`  |
| `--momentum`             | `0.9`  |
| `--wd`, `--weight-decay` | `1e-4` |
| `--lr-step-size`         | `30`   |
| `--lr-gamma`             | `0.1`  |

### AlexNet and VGG

Since `AlexNet` and the original `VGG` architectures do not include batch 
normalization, the default initial learning rate `--lr 0.1` is to high.

```
python main.py --model $MODEL --lr 1e-2
```

Here `$MODEL` is one of `alexnet`, `vgg11`, `vgg13`, `vgg16` or `vgg19`. Note
that `vgg11_bn`, `vgg13_bn`, `vgg16_bn`, and `vgg19_bn` include batch
normalization and thus are trained with the default parameters.

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


### MobileNetV3 Large & Small
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
     --model $MODEL --epochs 600 --opt rmsprop --batch-size 128 --lr 0.064\ 
     --wd 0.00001 --lr-step-size 2 --lr-gamma 0.973 --auto-augment imagenet --random-erase 0.2
```

Here `$MODEL` is one of `mobilenet_v3_large` or `mobilenet_v3_small`.

Then we averaged the parameters of the last 3 checkpoints that improved the Acc@1. See [#3182](https://github.com/pytorch/vision/pull/3182) 
and [#3354](https://github.com/pytorch/vision/pull/3354) for details.


## Mixed precision training
Automatic Mixed Precision (AMP) training on GPU for Pytorch can be enabled with the [NVIDIA Apex extension](https://github.com/NVIDIA/apex).

Mixed precision training makes use of both FP32 and FP16 precisions where appropriate. FP16 operations can leverage the Tensor cores on NVIDIA GPUs (Volta, Turing or newer architectures) for improved throughput, generally without loss in model accuracy. Mixed precision training also often allows larger batch sizes. GPU automatic mixed precision training for Pytorch Vision can be enabled via the flag value `--apex=True`.

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --model resnext50_32x4d --epochs 100 --apex
```

## Quantized

### Parameters used for generating quantized models:

For all post training quantized models (All quantized models except mobilenet-v2), the settings are:

1. num_calibration_batches: 32
2. num_workers: 16
3. batch_size: 32
4. eval_batch_size: 128
5. backend: 'fbgemm'

```
python train_quantization.py --device='cpu' --post-training-quantize --backend='fbgemm' --model='<model_name>'
```

For Mobilenet-v2, the model was trained with quantization aware training, the settings used are:
1. num_workers: 16
2. batch_size: 32
3. eval_batch_size: 128
4. backend: 'qnnpack'
5. learning-rate: 0.0001
6. num_epochs: 90
7. num_observer_update_epochs:4
8. num_batch_norm_update_epochs:3
9. momentum: 0.9
10. lr_step_size:30
11. lr_gamma: 0.1
12. weight-decay: 0.0001

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train_quantization.py --model='mobilenet_v2'
```

Training converges at about 10 epochs.

For Mobilenet-v3 Large, the model was trained with quantization aware training, the settings used are:
1. num_workers: 16
2. batch_size: 32
3. eval_batch_size: 128
4. backend: 'qnnpack'
5. learning-rate: 0.001
6. num_epochs: 90
7. num_observer_update_epochs:4
8. num_batch_norm_update_epochs:3
9. momentum: 0.9
10. lr_step_size:30
11. lr_gamma: 0.1
12. weight-decay: 0.00001

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train_quantization.py --model='mobilenet_v3_large' \
    --wd 0.00001 --lr 0.001
```

For post training quant, device is set to CPU. For training, the device is set to CUDA.

### Command to evaluate quantized models using the pre-trained weights:

```
python train_quantization.py --device='cpu' --test-only --backend='<backend>' --model='<model_name>'
```

