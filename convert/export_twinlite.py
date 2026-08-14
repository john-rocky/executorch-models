"""TwinLiteNet (drivable area + lane) -> ExecuTorch XNNPACK .pte. Stock ConvTranspose2d kept."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPOS = os.environ.get("CONVERT_REPOS", "")
sys.path.insert(0, os.path.join(REPOS, "TwinLiteNet"))
import torch
from harness import convert_and_gate
from model.TwinLite import TwinLiteNet as Net

net = Net()
sd = torch.load(os.path.join(REPOS, "TwinLiteNet", "pretrained", "best.pth"), map_location="cpu")
sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
print("load:", net.load_state_dict(sd, strict=False))

from calib import calib_loader

batches = calib_loader("street", (360, 640), "01")
cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "twinlitenet",
        net.eval(),
        (torch.randn(1, 3, 360, 640),),
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        # PReLU: XNNPACK 1.4.0 segfaults at execute (fix #21480 not in stable).
        exclude_configs=("PreluConfig",),
        # int8 does not run yet: the delegate fails XNNPACK shape propagation at
        # execute. Unresolved — excluding each partitioner config in turn does not
        # help, and minimal repros of this model's unusual ops (dilated, grouped,
        # depthwise-dilated conv, ConvTranspose2d with output_padding, quantized
        # slice) all pass on their own. conv/linear-only annotation is kept
        # because it rules the data-movement ops out as the cause.
        int8_op_types=[torch.ops.aten.conv2d.default, torch.ops.aten.linear.default],
        extra_meta={
            "source": "chequanghuy/TwinLiteNet (pretrained/best.pth)",
            "license": "MIT",
            "preprocess": "RGB 0-1, 360x640",
            "outputs": "drivable area [1,2,360,640] + lane line [1,2,360,640]",
        },
    )
