export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate "env$PYTHON_VERSION"
pip install C:/Users/circleci/project/dist/*.whl
cd c:/
python -c "import torch;import torchvision;print('Is torchvision useable?', all(x is not None for x in [torch.ops.image.decode_png, torch.ops.torchvision.roi_align]))"
