# Video Classification

TODO: Add some info about the context, dataset we use etc

## Data preparation

If you already have downloaded [Kinetics400 dataset](https://deepmind.com/research/open-source/kinetics), 
please proceed directly to the next section.

To download videos, one can use https://github.com/Showmax/kinetics-downloader

## Training

We assume the training and validation AVI videos are stored at `/data/kinectics400/train` and 
`/data/kinectics400/val`. 

### Multiple GPUs

Run the training on a single node with 8 GPUs:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path=/data/kinectics400 --train-dir=train --val-dir=val --batch-size=16 --cache-dataset --sync-bn --apex
```



### Single GPU 

**Note:** training on a single gpu can be extremely slow. 


```bash
python train.py --data-path=/data/kinectics400 --train-dir=train --val-dir=val --batch-size=8 --cache-dataset
```


