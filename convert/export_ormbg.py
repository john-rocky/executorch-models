"""ormbg (open background removal, ISNet) -> ExecuTorch XNNPACK .pte."""
import sys, os, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from harness import convert_and_gate

code_path = hf_hub_download("schirrmacher/ormbg", "ormbg/models/ormbg.py")
ckpt_path = hf_hub_download("schirrmacher/ormbg", "models/ormbg.pth")
spec = importlib.util.spec_from_file_location("ormbg_model", code_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

net = mod.ORMBG()
net.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
net.eval()


class Wrap(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return self.n(x)[0][0]  # sigmoid(d1) main mask


from calib import calib_loader

batches = calib_loader("portrait", 1024, "01")
cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "ormbg_isnet",
        Wrap(net).eval(),
        (torch.rand(1, 3, 1024, 1024),),
        runs=5,
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        extra_meta={
            "source": "schirrmacher/ormbg",
            "license": "Apache-2.0",
            "preprocess": "RGB 0-1, 1024x1024",
            "outputs": "alpha mask [1,1,1024,1024] 0-1 (sigmoid)",
        },
    )
