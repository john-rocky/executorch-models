"""RTMPose-s (2D body pose, 17 COCO keypoints) -> ExecuTorch XNNPACK .pte.

The model is built from mmpose's own CSPNeXt and RTMCCHead rather than a
reimplementation: hand-writing a backbone gives a strict state_dict load that
proves the shapes match and nothing about whether the forward is right, and there
would be no reference to check against.

mmpose needs mmengine and mmcv to import. mmcv-lite is enough — the three modules
that stop the import (xtcocotools, mmdet, mmcv._ext) are only reached by dataset
code, detection heads and compiled ops that CSPNeXt and RTMCCHead never call, so
they are stubbed rather than installed.

Output is SimCC: two 1-D coordinate distributions per keypoint rather than a
heatmap, which is what makes this architecture cheap. Decoding is an argmax per
axis, spelled out on the card.
"""
import sys, os, types, importlib.abc, importlib.machinery
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPOS = os.environ.get("CONVERT_REPOS", "")
sys.path.insert(0, os.path.join(REPOS, "mmpose"))


class _Stub(types.ModuleType):
    __path__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (object,), {})


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    PREFIXES = ("xtcocotools", "mmdet", "mmcv._ext")

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in self.PREFIXES):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _Stub(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _StubFinder())

import urllib.request
import tempfile
import torch
import torch.nn as nn
from harness import convert_and_gate
from calib import calib_loader
from mmpose.models.backbones import CSPNeXt
from mmpose.models.heads import RTMCCHead

URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
       "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth")
H, W = 256, 192
CKPT = os.path.join(os.path.expanduser("~/.cache/executorch-convert"), "rtmpose_s.pth")


class RTMPose(nn.Module):
    """backbone -> SimCC head. Config values are the ones in the checkpoint's own
    training config (rtmpose-s, 17 keypoints, 2x simcc split)."""

    def __init__(self):
        super().__init__()
        self.backbone = CSPNeXt(
            arch="P5", expand_ratio=0.5, deepen_factor=0.33, widen_factor=0.5,
            out_indices=(4,), channel_attention=True,
            norm_cfg=dict(type="SyncBN"), act_cfg=dict(type="SiLU"))
        self.head = RTMCCHead(
            in_channels=512, out_channels=17, input_size=(W, H),
            in_featuremap_size=(W // 32, H // 32), simcc_split_ratio=2.0,
            final_layer_kernel_size=7,
            gau_cfg=dict(hidden_dims=256, s=128, expansion_factor=2,
                         dropout_rate=0., drop_path=0., act_fn="SiLU",
                         use_rel_bias=False, pos_enc=False))

    def forward(self, x):
        x, y = self.head(self.backbone(x))
        return x.contiguous(), y.contiguous()


os.makedirs(os.path.dirname(CKPT), exist_ok=True)
if not os.path.exists(CKPT):
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, CKPT)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)

net = RTMPose()
missing, unexpected = net.load_state_dict(sd, strict=False)
assert not missing and not unexpected, f"missing={missing[:5]} unexpected={unexpected[:5]}"
net.eval()

# Person crops, not scenes. RTMPose is a top-down model: it expects a box around
# one person, and on whole photographs the fp32 build produces no coherent pose at
# all, so calibrating there would fit activations the model never sees in use.
# `convert/make_person_crops.py` builds the set by running the shipped YOLOX over
# the street and portrait photos.
batches = calib_loader("person", (H, W), "imagenet", n=12)
cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "rtmpose_s_body", net, (torch.randn(1, 3, H, W),),
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        # CSPNeXt's channel-attention block makes XNNPACKQuantizer's
        # adaptive_avg_pool2d annotator trip ("getitem_2 is not an aten
        # adaptive_avg_pool2d operator"). Annotating conv/linear only steps
        # around it.
        int8_op_types=[torch.ops.aten.conv2d.default, torch.ops.aten.linear.default],
        extra_meta={
            "source": "open-mmlab/mmpose RTMPose-s body7 "
                      "(rtmpose-s_simcc-body7_pt-body7_420e-256x192)",
            "license": "Apache-2.0",
            "preprocess": f"RGB, ImageNet norm, {H}x{W} person crop (detect first, "
                          f"then crop and resize to this aspect)",
            "outputs": f"SimCC pair: x [1,17,{W * 2}] and y [1,17,{H * 2}] — a 1-D "
                       f"distribution per keypoint per axis. Decode: keypoint k sits "
                       f"at (argmax(x[k]) / 2, argmax(y[k]) / 2) in crop pixels; the "
                       f"max value doubles as the confidence.",
        },
    )
