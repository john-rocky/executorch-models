"""MODNet (portrait matting) -> ExecuTorch XNNPACK .pte. Stock model, no GPU patches."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPOS = os.environ.get("CONVERT_REPOS", "")
sys.path.insert(0, os.path.join(REPOS, "MODNet"))
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from harness import convert_and_gate
from src.models.modnet import MODNet


class Matte(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x, True)[2]  # matte [1,1,512,512]


ckpt = hf_hub_download("DavG25/modnet-pretrained-models",
                       "models/modnet_photographic_portrait_matting.ckpt")
m = MODNet(backbone_pretrained=False).eval()
sd = {k.replace("module.", ""): v for k, v in
      torch.load(ckpt, map_location="cpu", weights_only=True).items()}
m.load_state_dict(sd)

from calib import calib_loader

batches = calib_loader("portrait", 512, "pm1")
cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "modnet_portrait_matting",
        Matte(m),
        (torch.randn(1, 3, 512, 512),),
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],  # parity on a real portrait (int8 clips randn)
        # quantized slice breaks XNNPACK shape-prop at execute (1.4.0) — keep
        # the split/slice ops on portable for int8
        exclude_configs=("SliceCopyConfig",) if prec == "int8" else None,
        extra_meta={
            "source": "ZHKKKe/MODNet + DavG25/modnet-pretrained-models ckpt",
            "license": "Apache-2.0",
            "preprocess": "RGB [-1,1], 512x512",
            "outputs": "alpha matte [1,1,512,512] 0-1",
        },
    )
