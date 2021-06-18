# Video Classification

We present a simple training script that can be used for replicating the result of [resenet-based video models](https://research.fb.com/wp-content/uploads/2018/04/a-closer-look-at-spatiotemporal-convolutions-for-action-recognition.pdf). All models are trained on [Kinetics400 dataset](https://deepmind.com/research/open-source/kinetics), a benchmark dataset for human-action recognition. The accuracy is reported on the traditional validation split.

## Data preparation

If you already have downloaded [Kinetics400 dataset](https://deepmind.com/research/open-source/kinetics), 
please proceed directly to the next section.

To download videos, one can use https://github.com/Showmax/kinetics-downloader. Please note that the dataset can take up upwards of 400GB, depending on the quality setting during download.

## Training

We assume the training and validation AVI videos are stored at `/data/kinectics400/train` and 
`/data/kinectics400/val`. For training we suggest starting with the hyperparameters reported in the [paper](https://research.fb.com/wp-content/uploads/2018/04/a-closer-look-at-spatiotemporal-convolutions-for-action-recognition.pdf), in order to match the performance of said models. Clip sampling strategy is a particularly important parameter during training, and we suggest using random temporal jittering during training - in other words sampling multiple training clips from each video with random start times during at every epoch. This functionality is built into our training script, and optimal hyperparameters are set by default.  

### Multiple GPUs

Run the training on a single node with 8 GPUs:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path=/data/kinectics400 --train-dir=train --val-dir=val --batch-size=16 --cache-dataset --sync-bn --apex
```

**Note:** all our models were trained on 8 nodes with 8 V100 GPUs each for a total of 64 GPUs. Expected training time for 64 GPUs is 24 hours, depending on the storage solution.
**Note 2:** hyperparameters for exact replication of our training can be found [here](https://github.com/pytorch/vision/blob/master/torchvision/models/video/README.md). Some hyperparameters such as learning rate are scaled linearly in proportion to the number of GPUs.

### Single GPU 

**Note:** training on a single gpu can be extremely slow. 


```bash
python train.py --data-path=/data/kinectics400 --train-dir=train --val-dir=val --batch-size=8 --cache-dataset
```
