"""PIDNet-S (Cityscapes semseg) -> ExecuTorch XNNPACK .pte. Weights via ONNX mirror."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPOS = os.environ.get("CONVERT_REPOS", "")
sys.path.insert(0, os.path.join(REPOS, "PIDNet"))
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper
from huggingface_hub import hf_hub_download
from harness import convert_and_gate
import models.pidnet as P

onnx_path = hf_hub_download("oenpu/PIDNet_S_enlight_friendly_onnx",
                            "PIDNet_S_enlight_friendly.onnx")
w = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy())
     for i in onnx.load(onnx_path).graph.initializer}

net = P.get_pred_model("pidnet_s", 19).eval()
sd = net.state_dict()
matched = {k: w[k] for k in sd if k in w}
assert len(matched) == len(sd), f"only {len(matched)}/{len(sd)} weights matched"
net.load_state_dict(matched, strict=False)


class Wrap(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        o = self.n(x)
        return o[0] if isinstance(o, (list, tuple)) else o


from calib import calib_loader

batches = calib_loader("street", 1024, "imagenet")
cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "pidnet_s_cityscapes",
        Wrap(net).eval(),
        (torch.randn(1, 3, 1024, 1024),),
        runs=5,
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        extra_meta={
            "source": "XuJiacong/PIDNet + oenpu/PIDNet_S_enlight_friendly_onnx weights",
            "license": "MIT",
            "preprocess": "RGB, ImageNet norm, 1024x1024",
            "outputs": "class logits [1,19,128,128] (argmax + upsample in app)",
        },
    )
