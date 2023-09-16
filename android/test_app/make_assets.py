import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)

print(torch.__version__)

model = fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
    box_score_thresh=0.7,
    rpn_post_nms_top_n_test=100,
    rpn_score_thresh=0.4,
    rpn_pre_nms_top_n_test=150,
)

model.eval()
script_model = torch.jit.script(model)
opt_script_model = optimize_for_mobile(script_model)
opt_script_model.save("app/src/main/assets/frcnn_mnetv3.pt")
