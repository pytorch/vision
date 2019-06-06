if [[ "$OSTYPE" == "msys" ]]; then
    CUDA_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v$1"
else
    CUDA_DIR="/usr/local/cuda-$1"
fi

if ! ls "$CUDA_DIR"
then
    echo "folder $CUDA_DIR not found to switch"
fi

echo "Switching symlink to $CUDA_DIR"
mkdir -p /usr/local
rm -fr /usr/local/cuda
ln -s "$CUDA_DIR" /usr/local/cuda

if [[ "$OSTYPE" == "msys" ]]; then
    export CUDA_VERSION=`ls /usr/local/cuda/bin/cudart64*.dll | head -1 | tr '._' ' ' | cut -d ' ' -f2`
    export CUDNN_VERSION=`ls /usr/local/cuda/bin/cudnn64*.dll | head -1 | tr '._' ' ' | cut -d ' ' -f2`
else
    export CUDA_VERSION=$(ls /usr/local/cuda/lib64/libcudart.so.*|sort|tac | head -1 | rev | cut -d"." -f -3 | rev)
    export CUDNN_VERSION=$(ls /usr/local/cuda/lib64/libcudnn.so.*|sort|tac | head -1 | rev | cut -d"." -f -3 | rev)
fi

ls -alh /usr/local/cuda

echo "CUDA_VERSION=$CUDA_VERSION"
echo "CUDNN_VERSION=$CUDNN_VERSION"
