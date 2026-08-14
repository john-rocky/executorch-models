"""RT-DETRv2-S (r18vd) -> ExecuTorch XNNPACK .pte, raw DETR head (no NMS needed).

DETR-style output: 300 queries with class logits + normalized cxcywh boxes.
App-side postprocess is sigmoid + top-k only (no NMS) — spec goes on the card.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import convert_and_gate
from transformers import RTDetrV2ForObjectDetection


class RawHead(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(pixel_values=x)
        return out.logits, out.pred_boxes  # [1,300,80], [1,300,4] cxcywh in 0..1


CKPT = "PekingU/rtdetr_v2_r18vd"
model = RTDetrV2ForObjectDetection.from_pretrained(CKPT)
x = torch.randn(1, 3, 640, 640)
from calib import calib_loader

batches = calib_loader("general", 640, "01")
cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "rtdetrv2_s_r18vd", RawHead(model), (x,), strip_asserts=True,
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        extra_meta={
            "source": CKPT,
            "license": "Apache-2.0",
            "preprocess": "RGB/255 only (no mean/std norm), 640x640",
            "outputs": "logits [1,300,80] (sigmoid -> per-class score), boxes [1,300,4] "
                       "cxcywh normalized 0..1; postprocess = sigmoid + top-k, NO NMS",
        },
    )
