"""D-FINE-S -> ExecuTorch XNNPACK .pte, raw DETR head (no NMS needed).

Same output contract as RT-DETRv2: 300 queries, class logits + normalized
cxcywh boxes; postprocess = sigmoid + top-k, no NMS.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import convert_and_gate
from transformers import DFineForObjectDetection


class RawHead(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(pixel_values=x)
        return out.logits, out.pred_boxes


CKPT = "ustc-community/dfine-small-coco"
model = DFineForObjectDetection.from_pretrained(CKPT)
x = torch.randn(1, 3, 640, 640)
convert_and_gate(
    "dfine_s_coco", RawHead(model), (x,), strip_asserts=True,
    extra_meta={
        "source": CKPT,
        "license": "Apache-2.0",
        "preprocess": "RGB/255 only (no mean/std norm), 640x640",
        "outputs": "logits [1,300,80] (sigmoid -> per-class score), boxes [1,300,4] "
                   "cxcywh normalized 0..1; postprocess = sigmoid + top-k, NO NMS",
    },
)
