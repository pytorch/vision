#!/bin/bash
#SBATCH --partition=train
#SBATCH --cpus-per-task=96  # 12 CPUs per GPU
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --time=70:00:00


# echo mobilenet_v3_large
# torchrun --nproc_per_node=8 train.py  --model mobilenet_v3_large --workers 12 --weights MobileNet_V3_Large_Weights.DEFAULT --test-only 
# torchrun --nproc_per_node=8 train.py  --model mobilenet_v3_large --workers 12 --weights MobileNet_V3_Large_Weights.DEFAULT --test-only --convert-to-tensor-first
# torchrun --nproc_per_node=8 train.py  --model mobilenet_v3_large --workers 12 --weights MobileNet_V3_Large_Weights.DEFAULT --test-only --convert-to-tensor-first --antialias

# echo resnet50
# torchrun --nproc_per_node=8 train.py  --model resnet50 --workers 12 --weights ResNet50_Weights.DEFAULT --test-only 
# torchrun --nproc_per_node=8 train.py  --model resnet50 --workers 12 --weights ResNet50_Weights.DEFAULT --test-only --convert-to-tensor-first 
# torchrun --nproc_per_node=8 train.py  --model resnet50 --workers 12 --weights ResNet50_Weights.DEFAULT --test-only --convert-to-tensor-first --antialias

# echo vit
# torchrun --nproc_per_node=8 train.py  --model vit_b_16 --workers 12 --weights ViT_B_16_Weights.DEFAULT --test-only 
# torchrun --nproc_per_node=8 train.py  --model vit_b_16 --workers 12 --weights ViT_B_16_Weights.DEFAULT --test-only --convert-to-tensor-first 
# torchrun --nproc_per_node=8 train.py  --model vit_b_16 --workers 12 --weights ViT_B_16_Weights.DEFAULT --test-only --convert-to-tensor-first --antialias

echo efficientnet_b4
torchrun --nproc_per_node=8 train.py  --model efficientnet_b4 --workers 12 --weights EfficientNet_B4_Weights.DEFAULT --test-only
torchrun --nproc_per_node=8 train.py  --model efficientnet_b4 --workers 12 --weights EfficientNet_B4_Weights.DEFAULT --test-only --convert-to-tensor-first 
torchrun --nproc_per_node=8 train.py  --model efficientnet_b4 --workers 12 --weights EfficientNet_B4_Weights.DEFAULT --test-only --convert-to-tensor-first --antialias

echo shufflenet_v2_x1_0
torchrun --nproc_per_node=8 train.py  --model shufflenet_v2_x1_0 --workers 12 --weights ShuffleNet_V2_X1_0_Weights.DEFAULT --test-only
torchrun --nproc_per_node=8 train.py  --model shufflenet_v2_x1_0 --workers 12 --weights ShuffleNet_V2_X1_0_Weights.DEFAULT --test-only --convert-to-tensor-first 
torchrun --nproc_per_node=8 train.py  --model shufflenet_v2_x1_0 --workers 12 --weights ShuffleNet_V2_X1_0_Weights.DEFAULT --test-only --convert-to-tensor-first --antialias

