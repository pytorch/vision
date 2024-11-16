# =========================================================
# BEGIN REPRO SCRIPT
# =========================================================
import torch
from torch.testing._internal.optests import opcheck

# Make sure you have loaded the library that contains the op
# via an import or torch.ops.load_library(...)
# op = torch.ops.torchvision.deform_conv2d.default
op = torch.ops.torchvision.roi_align.default
args, kwargs = torch.load("/var/folders/m7/m4jyvbb97ml6nftpw7b6fsk00000gn/T/pytorch_opcheck_safe_to_delete/repro_173109241941725.22.pt")
opcheck(op, args, kwargs, test_utils="test_autograd_registration")
# =========================================================
# END REPRO SCRIPT
# =========================================================