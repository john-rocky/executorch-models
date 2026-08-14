"""U^2-Net (salient object segmentation) -> ExecuTorch XNNPACK .pte.

Weights are Carve's universal retrain of the stock U2NET architecture, which is
the Apache-2.0 route — the original repo publishes its checkpoints on Google
Drive. Six side outputs again; only the main one ships.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPOS = os.environ.get("CONVERT_REPOS", "")
sys.path.insert(0, os.path.join(REPOS, "U-2-Net"))
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from harness import convert_and_gate
from calib import calib_loader
from model.u2net import U2NET

ckpt = hf_hub_download("Carve/u2net-universal", "full_weights.pth")
net = U2NET(3, 1)
net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
net.eval()


class Wrap(nn.Module):
    """U2NET.forward already applies the sigmoid to each side output; keep d0."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return self.n(x)[0]


batches = calib_loader("general", 320, "imagenet")
cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "u2net", Wrap(net).eval(), (torch.rand(1, 3, 320, 320),),
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        extra_meta={
            "source": "xuebinqin/U-2-Net + Carve/u2net-universal weights",
            "license": "Apache-2.0",
            "preprocess": "RGB, ImageNet norm, 320x320",
            "outputs": "saliency mask [1,1,320,320] 0-1 (sigmoid); "
                       "min-max normalize then resize to the source image",
        },
    )
