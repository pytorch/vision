#!/bin/bash

#SBATCH --mail-type=ALL                     # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=./%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=./%j.err                  # where to store error messages
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --nodelist=tikgpu06

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID

# exit on errors
set -o errexit
# execute python file
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --test-only --pretrained
echo finished at: `date`
exit 0;




