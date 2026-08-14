"""DINOv2 ViT-S/14 -> ExecuTorch XNNPACK .pte (cls token + patch tokens)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import convert_and_gate


class Dinov2Features(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        feats = self.m.forward_features(x)
        return feats["x_norm_clstoken"], feats["x_norm_patchtokens"]


m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
x = torch.randn(1, 3, 518, 518)
from calib import calib_loader

batches = calib_loader("general", 518, "imagenet")
cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "dinov2_vits14",
        Dinov2Features(m),
        (x,),
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        int8_dynamic=True,  # ViT: static int8 wrecks features (corr 0.38); dynamic holds
        extra_meta={
            "source": "facebookresearch/dinov2 (torch.hub)",
            "license": "Apache-2.0",
            "preprocess": "RGB, ImageNet norm, 518x518",
            "outputs": "cls token [1,384], patch tokens [1,1369,384]",
        },
    )
