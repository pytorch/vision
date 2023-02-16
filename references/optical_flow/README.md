# Optical flow reference training scripts

This folder contains reference training scripts for optical flow.
They serve as a log of how to train specific models, so as to provide baseline
training and evaluation scripts to quickly bootstrap research.


### RAFT Large

The RAFT large model was trained on Flying Chairs and then on Flying Things.
Both used 8 A100 GPUs and a batch size of 2 (so effective batch size is 16). The
rest of the hyper-parameters are exactly the same as the original RAFT training
recipe from https://github.com/princeton-vl/RAFT. The original recipe trains for
100000 updates (or steps) on each dataset - this corresponds to about 72 and 20
epochs on Chairs and Things respectively:

```
num_epochs = ceil(num_steps / number_of_steps_per_epoch)
           = ceil(num_steps / (num_samples / effective_batch_size))
```

```
torchrun --nproc_per_node 8 --nnodes 1 train.py \
    --dataset-root $dataset_root \
    --name $name_chairs \
    --model raft_large \
    --train-dataset chairs \
    --batch-size 2 \
    --lr 0.0004 \
    --weight-decay 0.0001 \
    --epochs 72 \
    --output-dir $chairs_dir
```

```
torchrun --nproc_per_node 8 --nnodes 1 train.py \
    --dataset-root $dataset_root \
    --name $name_things \
    --model raft_large \
    --train-dataset things \
    --batch-size 2 \
    --lr 0.000125 \
    --weight-decay 0.0001 \
    --epochs 20 \
    --freeze-batch-norm \
    --output-dir $things_dir\
    --resume $chairs_dir/$name_chairs.pth
```


### Evaluation

```
torchrun --nproc_per_node 1 --nnodes 1 train.py --val-dataset sintel --batch-size 1 --dataset-root $dataset_root --model raft_large --weights Raft_Large_Weights.C_T_SKHT_V2
```

This should give an epe of about 1.3822 on the clean pass and 2.7161 on the
final pass of Sintel-train. Results may vary slightly depending on the batch
size and the number of GPUs. For the most accurate results use 1 GPU and
`--batch-size 1`:

```
Sintel val clean epe: 1.3822	1px: 0.9028	3px: 0.9573	5px: 0.9697	per_image_epe: 1.3822	f1: 4.0248
Sintel val final epe: 2.7161	1px: 0.8528	3px: 0.9204	5px: 0.9392	per_image_epe: 2.7161	f1: 7.5964
```

You can also evaluate on Kitti train:

```
torchrun --nproc_per_node 1 --nnodes 1 train.py --val-dataset kitti --batch-size 1 --dataset-root $dataset_root --model raft_large --weights Raft_Large_Weights.C_T_SKHT_V2
Kitti val epe: 4.7968	1px: 0.6388	3px: 0.8197	5px: 0.8661	per_image_epe: 4.5118	f1: 16.0679
```
