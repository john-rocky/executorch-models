"""DIS (IS-Net, dichotomous image segmentation) -> ExecuTorch XNNPACK .pte.

Same IS-Net architecture as ormbg, trained for high-accuracy object cutouts with
fine structures (hair, wires, mesh) rather than background removal. The model
returns 6 side outputs; only the main one is exported, matching the ormbg card.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPOS = os.environ.get("CONVERT_REPOS", "")
sys.path.insert(0, os.path.join(REPOS, "DIS", "IS-Net"))
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from harness import convert_and_gate
from calib import calib_loader
from models.isnet import ISNetDIS

ckpt = hf_hub_download("NimaBoscarino/IS-Net_DIS-general-use", "isnet-general-use.pth")
net = ISNetDIS()
net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
net.eval()


class Wrap(nn.Module):
    """IS-Net returns ([d1..d6], [feature maps]); take the main side output.

    Do not add a sigmoid: the stock forward already returns `F.sigmoid(d1)`. An
    extra one squashes the mask into [0.5, 0.731], where thresholding at 0.5 marks
    the whole image as foreground — and every parity number stays at 1.000000
    because the exported graph faithfully reproduces the wrapper it was given."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return self.n(x)[0][0]


batches = calib_loader("general", 1024, "pm1")
cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "dis_isnet", Wrap(net).eval(), (torch.rand(1, 3, 1024, 1024),), runs=5,
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        extra_meta={
            "source": "xuebinqin/DIS + NimaBoscarino/IS-Net_DIS-general-use weights",
            "license": "Apache-2.0",
            "preprocess": "RGB, scaled to [-1,1] (x/255 then (x-0.5)/0.5), 1024x1024",
            "outputs": "alpha mask [1,1,1024,1024] 0-1 (sigmoid)",
        },
    )
