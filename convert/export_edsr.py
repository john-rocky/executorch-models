"""EDSR-base x4 -> ExecuTorch XNNPACK .pte (stock model; PixelShuffle exports fine on CPU)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import convert_and_gate
from super_image import EdsrModel

m = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=4).eval()
x = torch.rand(1, 3, 128, 128)
from calib import calib_loader

batches = calib_loader("general", 128, "01")
cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "edsr_base_x4",
        m,
        (x,),
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        extra_meta={
            "source": "eugenesiow/edsr-base (super-image)",
            "license": "Apache-2.0",
            "preprocess": "RGB 0-1, 128x128 tile",
            "outputs": "SR image [1,3,512,512] RGB 0-1",
        },
    )
