"""Real-ESRGAN general x4v3 (SRVGGNetCompact) -> ExecuTorch XNNPACK .pte.

Weights come from the upstream release asset, which is the BSD-3-Clause route;
the copies on the Hub either carry no license or are already-converted formats.
The architecture is written out here rather than imported: the vendored
`srvgg_arch.py` pulls in basicsr for nothing but a registry decorator, and
loading the checkpoint with strict=True is the real check that this matches.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from harness import convert_and_gate
from calib import calib_loader

URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
       "realesr-general-x4v3.pth")
CACHE = os.path.join(os.path.expanduser("~/.cache/executorch-convert"),
                     "realesr-general-x4v3.pth")


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # the network predicts the residual over a nearest-neighbour upscale
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


os.makedirs(os.path.dirname(CACHE), exist_ok=True)
if not os.path.exists(CACHE):
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, CACHE)
sd = torch.load(CACHE, map_location="cpu", weights_only=True)
sd = sd.get("params", sd)

net = SRVGGNetCompact()
net.load_state_dict(sd)  # strict: proves the architecture above is the right one
net.eval()

batches = calib_loader("general", 128, "01")
cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "real_esrgan_x4v3", net, (torch.rand(1, 3, 128, 128),),
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        # PReLU: XNNPACK 1.4.0 segfaults at execute (upstream fix #21480 is not
        # in the stable wheel). This model is PReLU on every other layer.
        exclude_configs=("PreluConfig",),
        # Annotating only conv keeps the PixelShuffle reshapes out of
        # quantization; letting them in fails XNNPACK shape propagation at
        # execute ("Propagating input shapes failed").
        int8_op_types=[torch.ops.aten.conv2d.default],
        extra_meta={
            "source": "xinntao/Real-ESRGAN release v0.2.5.0 (realesr-general-x4v3)",
            "license": "BSD-3-Clause",
            "preprocess": "RGB 0-1, 128x128 tile",
            "outputs": "SR image [1,3,512,512] RGB 0-1",
        },
    )
