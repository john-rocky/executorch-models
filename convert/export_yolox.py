"""YOLOX-s (Apache-2.0) -> ExecuTorch XNNPACK .pte, decoded head (NMS app-side).

AGPL YOLO11/26 can't be shipped on HF; YOLOX is the Apache detection fallback.
Loaded via torch.hub from Megvii-BaseDetection/YOLOX. Export keeps the head's
grid decode (pure tensor math) so the output is [1,8400,85] = (cx,cy,w,h in
input pixels, objectness, 80 class scores). App postprocess: obj*cls threshold
+ NMS — unlike the DETR pair, NMS IS required here; spec goes on the card.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import convert_and_gate

# trust_repo: headless run — torch.hub's interactive trust prompt EOFs otherwise
model = torch.hub.load("Megvii-BaseDetection/YOLOX", "yolox_s", pretrained=True,
                       trust_repo=True)
model.head.decode_in_inference = True


class Decoded(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)  # [1, 8400, 85]


x = torch.randn(1, 3, 640, 640)
convert_and_gate(
    "yolox_s", Decoded(model), (x,),
    extra_meta={
        "source": "Megvii-BaseDetection/YOLOX (yolox_s)",
        "license": "Apache-2.0",
        "preprocess": "BGR 0..255 float, NO normalization (YOLOX v0.3+ convention), 640x640 letterbox pad 114",
        "outputs": "[1,8400,85]: cx,cy,w,h (input px), objectness, 80 class scores; "
                   "postprocess = obj*cls threshold + NMS (required)",
    },
)
