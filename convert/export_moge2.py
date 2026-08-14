"""MoGe-2 ViT-S (monocular geometry: point map, normals, mask, metric scale)
-> ExecuTorch XNNPACK .pte.

One forward gives an app four things at once — a metric point map, surface normals,
a validity mask, and the scale factor — which is why this is worth having on device
even though a plain depth model is smaller.

`MoGeModel.forward` derives its token grid from `num_tokens` at runtime and returns
a dict. Both are fixed here: a static 518x518 input at 1369 tokens, and a wrapper
that returns the four tensors in order (the harness takes tensors, not dicts).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPOS = os.environ.get("CONVERT_REPOS", "")
sys.path.insert(0, os.path.join(REPOS, "MoGe"))
import types
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from harness import convert_and_gate
from calib import calib_loader

# `moge.model.v2` imports utils3d at module scope but only calls it inside
# `infer()`, which this export does not touch. utils3d depends on open3d, which
# has no CPython 3.12 wheel, so stub it rather than pin the environment to 3.11.
_u3d = types.ModuleType("utils3d")
_u3d.pt = types.SimpleNamespace()
sys.modules.setdefault("utils3d", _u3d)

from moge.model.v2 import MoGeModel

CKPT = "Ruicheng/moge-2-vits-normal"
SIZE = 518
# 518/14 = 37 patches per side; the model's own range is 1200-3600 tokens.
NUM_TOKENS = (SIZE // 14) ** 2

model = MoGeModel.from_pretrained(hf_hub_download(CKPT, "model.pt")).eval()


class Geometry(nn.Module):
    """points, normal, mask, metric_scale — always in that order, always present."""

    def __init__(self, m, num_tokens):
        super().__init__()
        self.m = m
        self.num_tokens = num_tokens

    def forward(self, image):
        out = self.m(image, self.num_tokens)
        return (out["points"].contiguous(),
                out["normal"].contiguous(),
                out["mask"].contiguous(),
                out["metric_scale"].contiguous())


net = Geometry(model, NUM_TOKENS).eval()
batches = calib_loader("general", SIZE, "imagenet")
with torch.no_grad():
    ref = net(batches[0][0])
print("outputs:", [tuple(t.shape) for t in ref])

cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "moge2_vits", net, (torch.randn(1, 3, SIZE, SIZE),), runs=5,
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        int8_dynamic=True,  # DINOv2 backbone: static int8 wrecks attention
        extra_meta={
            "source": CKPT,
            "license": "MIT",
            "preprocess": f"RGB, ImageNet norm, {SIZE}x{SIZE}",
            "outputs": "points [1,H,W,3] metric point map, normal [1,H,W,3], "
                       "mask [1,H,W] validity, metric_scale [1]",
        },
    )
