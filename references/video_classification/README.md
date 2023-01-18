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
torchrun --nproc_per_node=8 train.py --data-path=/data/kinectics400 --kinetics-version="400" --lr 0.08 --cache-dataset --sync-bn --amp 
```

**Note:** all our models were trained on 8 nodes with 8 V100 GPUs each for a total of 64 GPUs. Expected training time for 64 GPUs is 24 hours, depending on the storage solution.
**Note 2:** hyperparameters for exact replication of our training can be found on the section below. Some hyperparameters such as learning rate must be scaled linearly in proportion to the number of GPUs. The default values assume 64 GPUs.

### Single GPU 

**Note:** training on a single gpu can be extremely slow. 


```bash
python train.py --data-path=/data/kinectics400 --kinetics-version="400" --batch-size=8 --cache-dataset
```


### Additional Kinetics versions

Since the original release, additional versions of Kinetics dataset became available (Kinetics 600).
Our training scripts support these versions of dataset as well by setting the `--kinetics-version` parameter to `"600"`.

**Note:** training on Kinetics 600 requires a different set of hyperparameters for optimal performance. We do not provide Kinetics 600 pretrained models.


## Video classification models

Starting with version `0.4.0` we have introduced support for basic video tasks and video classification modelling.
For more information about the available models check [here](https://pytorch.org/docs/stable/torchvision/models.html#video-classification). 

### Video ResNet models

See reference training script [here](https://github.com/pytorch/vision/blob/main/references/video_classification/train.py):

- input space: RGB
- resize size: [128, 171]
- crop size: [112, 112]
- mean: [0.43216, 0.394666, 0.37645]
- std: [0.22803, 0.22145, 0.216989]
- number of classes: 400

Input data augmentations at training time (with optional parameters):

1. ConvertImageDtype
2. Resize (resize size value above)
3. Random horizontal flip (0.5)
4. Normalization (mean, std, see values above)
5. Random Crop (crop size value above)
6. Convert BCHW to CBHW

Input data augmentations at validation time (with optional parameters):

1. ConvertImageDtype
2. Resize (resize size value above)
3. Normalization (mean, std, see values above)
4. Center Crop (crop size value above)
5. Convert BCHW to CBHW

This translates in the following set of command-line arguments. Please note that `--batch-size` parameter controls the
batch size per GPU. Moreover, note that our default `--lr` is configured for 64 GPUs which is how many we used for the 
Video resnet models:
```
# number of frames per clip
--clip_len 16 \ 
--frame-rate 15 \
# allow for temporal jittering
--clips_per_video 5 \
--batch-size 24 \
--epochs 45 \
--lr 0.64 \
# we use 10 epochs for linear warmup
--lr-warmup-epochs 10 \
# learning rate is decayed at 20, 30, and 40 epoch by a factor of 10
--lr-milestones 20, 30, 40 \
--lr-gamma 0.1 \
--train-resize-size 128 171 \
--train-crop-size 112 112 \
--val-resize-size 128 171 \
--val-crop-size 112 112
```

### S3D

The S3D model was trained similarly to the above but with the following changes on the default configuration:
```
--batch-size=12 --lr 0.2 --clip-len 64 --clips-per-video 5 --sync-bn \
--train-resize-size 256 256 --train-crop-size 224 224 --val-resize-size 256 256 --val-crop-size 224 224
```

We used 64 GPUs to train the architecture. 

To estimate the validation statistics of the model, we run the reference script with the following configuration:
```
--batch-size=16 --test-only --clip-len 128 --clips-per-video 1 
```

### Additional video modelling resources

- [Video Model Zoo](https://github.com/facebookresearch/VMZ)
- [PySlowFast](https://github.com/facebookresearch/SlowFast)

### References

[0] _D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun and M. Paluri_: A Closer Look at Spatiotemporal Convolutions for Action Recognition. _CVPR 2018_ ([paper](https://research.fb.com/wp-content/uploads/2018/04/a-closer-look-at-spatiotemporal-convolutions-for-action-recognition.pdf))

[1] _W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan, F. Viola, T. Green, T. Back, P. Natsev, M. Suleyman, A. Zisserman_: The Kinetics Human Action Video Dataset ([paper](https://arxiv.org/abs/1705.06950))
