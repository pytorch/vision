# Optical flow reference training scripts

This folder contains reference training scripts for optical flow.
They serve as a log of how to train specific models, so as to provide baseline
training and evaluation scripts to quickly bootstrap research.


### RAFT Large

The RAFT large model was trained on Flying Chairs and then on Flying Things.
Both used 8 A100 GPUs and a batch size of 2 (so effective batch size is 16). The
rest of the hyper-parameters are exactly the same as the original RAFT training
recipe from https://github.com/princeton-vl/RAFT.

```
torchrun --nproc_per_node 8 --nnodes 1 train.py \
    --dataset-root $dataset_root \
    --name $name_chairs \
    --train-dataset chairs \
    --batch-size 2 \
    --lr 0.0004 \
    --weight-decay 0.0001 \
    --num-steps 100000 \
    --output-dir $chairs_dir
```

```
torchrun --nproc_per_node 8 --nnodes 1 train.py \
    --dataset-root $dataset_root \
    --name $name_things \
    --train-dataset things \
    --batch-size 2 \
    --lr 0.000125 \
    --weight-decay 0.0001 \
    --num-steps 100000 \
    --freeze-batch-norm \
    --output-dir $things_dir\
    --resume $chairs_dir/$name_chairs.pth
```


### Evaluation

```
torchrun --nproc_per_node 8 --nnodes 1 train.py --val-dataset sintel --batch-size 10 --dataset-root $dataset_root --model raft_large --pretrained
```

This should give an epe of about 1.3825 on the clean pass and 2.7148 on the
final pass of Sintel. Results may vary slightly depending on the batch size and
the number of GPUs. For the most accurate resuts use 1 GPU and `--batch-size 1`.

```
Sintel val clean epe: 1.3825	1px: 0.9028	3px: 0.9573	5px: 0.9697	per_image_epe: 1.3782	f1: 4.0234
Sintel val final epe: 2.7148	1px: 0.8526	3px: 0.9203	5px: 0.9392	per_image_epe: 2.7199	f1: 7.6100
```
