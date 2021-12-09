#!/bin/bash
#SBATCH --partition=train
#SBATCH --cpus-per-task=96  # 12 CPUs per GPU
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --time=70:00:00
#SBATCH --output=/data/home/nicolashug/cluster/experiments/slurm-%j.out
#SBATCH --error=/data/home/nicolashug/cluster/experiments/slurm-%j.err



n_gpus=8  # If you modify these, also update the equivalent above.
n_nodes=1

output_dir=~/cluster/experiments/id_$SLURM_JOB_ID
mkdir -p $output_dir

this_script=./train.sh  # depends where you call it from
cp $this_script $output_dir

function unused_port() {
    # Find a random unused port. It's needed if you run multiple sbatches on the same node
    N=${1:-1}
    comm -23 \
        <(seq "1025" "65535" | sort) \
        <(ss -Htan |
            awk '{print $4}' |
            cut -d':' -f2 |
            sort -u) |
        shuf |
        head -n "$N"
}
master_port=$(unused_port)

dataset_root=/data/home/nicolashug/cluster/work/downloads

# FlyingChairs
batch_size_chairs=2
lr_chairs=0.0004
epochs_chairs=72
name_chairs=raft_chairs
wdecay_chairs=0.0001

chairs_dir=$output_dir/chairs
mkdir -p $chairs_dir
torchrun --nproc_per_node $n_gpus --nnodes $n_nodes --master_port $master_port references/optical_flow/train.py \
    --dataset-root $dataset_root \
    --name $name_chairs \
    --train-dataset chairs \
    --batch-size $batch_size_chairs \
    --lr $lr_chairs \
    --weight-decay $wdecay_chairs \
    --epochs $epochs_chairs \
    --output-dir $chairs_dir

# FlyingThings3D
batch_size_things=2
lr_things=0.000125
epochs_things=20
name_things=raft_things
wdecay_things=0.0001

things_dir=$output_dir/things
mkdir -p $things_dir
torchrun --nproc_per_node $n_gpus --nnodes $n_nodes --master_port $master_port references/optical_flow/train.py \
    --dataset-root $dataset_root \
    --name $name_things \
    --train-dataset things \
    --batch-size $batch_size_things \
    --lr $lr_things \
    --weight-decay $wdecay_things \
    --epochs $epochs_things \
    --freeze-batch-norm \
    --output-dir $things_dir\
    --resume $chairs_dir/$name_chairs.pth

# Sintel S+K+H
batch_size_sintel_skh=2
lr_sintel_skh=0.000125
epochs_sintel_skh=6
name_sintel_skh=raft_sintel_skh
wdecay_sintel_skh=0.00001
gamma_sintel_skh=0.85

sintel_skh_dir=$output_dir/sintel_skh
mkdir -p $sintel_skh_dir
torchrun --nproc_per_node $n_gpus --nnodes $n_nodes --master_port $master_port references/optical_flow/train.py \
    --dataset-root $dataset_root \
    --name $name_sintel_skh \
    --train-dataset sintel_SKH \
    --batch-size $batch_size_sintel_skh \
    --lr $lr_sintel_skh \
    --weight-decay $wdecay_sintel_skh \
    --gamma $gamma_sintel_skh \
    --epochs $epochs_sintel_skh \
    --freeze-batch-norm \
    --output-dir $sintel_skh_dir\
    --resume $things_dir/$name_things.pth

# Kitti
batch_size_kitti=2
lr_kitti=0.0001
epochs_kitti=4000
name_kitti=raft_kitti
wdecay_kitti=0.00001
gamma_kitti=0.85

kitti_dir=$output_dir/kitti
mkdir -p $kitti_dir
torchrun --nproc_per_node $n_gpus --nnodes $n_nodes --master_port $master_port references/optical_flow/train.py \
    --dataset-root $dataset_root \
    --name $name_kitti \
    --train-dataset kitti \
    --batch-size $batch_size_kitti \
    --lr $lr_kitti \
    --weight-decay $wdecay_kitti \
    --gamma $gamma_kitti \
    --epochs $epochs_kitti \
    --freeze-batch-norm \
    --output-dir $kitti_dir \
    --resume $sintel_skh_dir/$name_sintel_skh.pth