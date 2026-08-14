"""SSDLite320-MobileNetV3-Large -> ExecuTorch XNNPACK .pte (raw head outputs).

Same 4D-head-tap as the LiteRT ship: 12 tensors, (cls, box) per FPN level;
anchor decode + NMS happen in app code.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from harness import convert_and_gate
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights)


class SSDRaw4D(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        feats = list(self.m.backbone(x).values())
        ch = self.m.head.classification_head.module_list
        rh = self.m.head.regression_head.module_list
        outs = []
        for i, f in enumerate(feats):
            outs.append(ch[i](f))   # [N, A*91, H, W]
            outs.append(rh[i](f))   # [N, A*4,  H, W]
        return tuple(outs)


m = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
x = torch.randn(1, 3, 320, 320)
from calib import calib_loader

# Street scenes, not landscapes — see the note in export_yolox.py.
batches = calib_loader("street", 320, "01")
cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "ssdlite320_mobilenetv3",
        SSDRaw4D(m),
        (x,),
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        extra_meta={
            "source": "torchvision ssdlite320_mobilenet_v3_large COCO_V1",
            "license": "BSD-3-Clause",
            "preprocess": "RGB 0-1, 320x320 (torchvision SSDLite norm baked in model)",
            "outputs": "12 raw heads: (cls [1,A*91,H,W], box [1,A*4,H,W]) x 6 levels, H=W in {20,10,5,3,2,1}",
        },
    )
