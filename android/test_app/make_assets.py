import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

print(torch.__version__)

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    pretrained=True,
    box_score_thresh=0.7,
    rpn_post_nms_top_n_test=100,
    rpn_score_thresh=0.4,
    rpn_pre_nms_top_n_test=150)

model.eval()
script_model = torch.jit.script(model)
opt_script_model = optimize_for_mobile(script_model)
opt_script_model.save("app/src/main/assets/frcnn_mnetv3.pt")
