#!/bin/bash
# Prerequisite: conda activate torch-xla-nightly-vision
#set -e

# TPU XLA
#export TPU_IP_ADDRESS="10.2.101.2"
#export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

# CPU XLA
#export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
#export XRT_WORKERS="localservice:0;grpc://localhost:40934"
export XRT_TPU_CONFIG="localservice;0;localhost:51011"

# Helps us but still buggy, still have to use padding a lot
export XLA_EXPERIMENTAL=nonzero:masked_select

# Enable/disable bfloat16
export XLA_USE_BF16=1
#export XLA_DOWNCAST_BF16=1

#TF_CPP_VMODULE=tensor=5,computation_client=5,xrt_computation_client=5,aten_xla_type=1 XLA_SAVE_TENSORS_FILE=/tmp/maskrcnn/graphs.hlo XLA_SAVE_TENSORS_FMT=hlo TPU_STDERR_LOG_LEVEL=0 
python3 ~/vision/references/detection/train_tpu.py "${@}"
