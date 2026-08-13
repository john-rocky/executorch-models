"""Depth-Anything-V2-Small -> ExecuTorch XNNPACK .pte."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import convert_and_gate
from transformers import AutoModelForDepthEstimation


class DepthWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).predicted_depth


m = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
x = torch.randn(1, 3, 518, 518)
convert_and_gate(
    "depth_anything_v2_small",
    DepthWrapper(m),
    (x,),
    extra_meta={
        "source": "depth-anything/Depth-Anything-V2-Small-hf",
        "license": "Apache-2.0",
        "preprocess": "RGB, ImageNet norm, 518x518",
        "outputs": "relative inverse depth [1,518,518]",
    },
)
