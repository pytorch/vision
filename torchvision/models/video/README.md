## Video classification models

Starting with version `0.4.0` we have introduced support for basic video tasks and video classification modelling.
At the moment, our pretraining consists of base implementation of popular resnet-based video models [0], together with their
basic variant pre-trained on Kinetics400 [1]. Although this is a standard benchmark pre-training, we are always considering what is the best for the community.

Additional documentation can be found [here](https://pytorch.org/docs/stable/torchvision/models.html#video-classification). 

### Kinetics400 dataset pretraining parameters

See reference training script [here](https://github.com/pytorch/vision/blob/master/references/video_classification/train.py):

- input size: [3, 16, 112, 112]
- input space: RGB
- input range: [0, 1]
- mean: [0.43216, 0.394666, 0.37645]
- std: [0.22803, 0.22145, 0.216989]
- number of classes: 400

Input data augmentations at training time (with optional parameters):

0. ToTensor
1. Resize (128, 171)
2. Random horizontal flip (0.5)
3. Normalization (mean, std, see values above)
4. Random Crop (112, 112)

Input data augmentations at validation time (with optional parameters):

0. ToTensor
1. Resize (128, 171)
2. Normalization (mean, std, see values above)
3. Center Crop (112, 112)

This translates in the following set of command-line arguments (please note that learning rate and batch size end up being scaled by the number of GPUs; all our models were trained on 8 nodes with 8 V100 GPUs each for a total of 64 GPUs):
```
# number of frames per clip
--clip_len 16 \ 
# allow for temporal jittering
--clips_per_video 5 \
--batch-size 24 \
--epochs 45 \
--lr 0.01 \
# we use 10 epochs for linear warmup
--lr-warmup-epochs 10 \
# learning rate is decayed at 20, 30, and 40 epoch by a factor of 10
--lr-milestones 20, 30, 40 \
--lr-gamma 0.1 
```

### Additional video modelling resources

- [Video Model Zoo](https://github.com/facebookresearch/VMZ)
- [PySlowFast](https://github.com/facebookresearch/SlowFast)

### References

[0] _D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun and M. Paluri_: A Closer Look at Spatiotemporal Convolutions for Action Recognition. _CVPR 2018_ ([paper](https://research.fb.com/wp-content/uploads/2018/04/a-closer-look-at-spatiotemporal-convolutions-for-action-recognition.pdf))

[1] _W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan, F. Viola, T. Green, T. Back, P. Natsev, M. Suleyman, A. Zisserman_: The Kinetics Human Action Video Dataset ([paper](https://arxiv.org/abs/1705.06950))