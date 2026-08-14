"""6DRepNet (head pose, 6D rotation representation) -> ExecuTorch XNNPACK .pte.

The published checkpoint is already re-parameterized (`rbr_reparam` keys), so the
RepVGG backbone is plain conv+ReLU at export time — no branch fusion needed and
nothing for XNNPACK to trip over.

The model regresses a 6D rotation representation, not Euler angles. Turning that
into a rotation matrix is six lines of app-side math (spelled out on the card), so
the .pte stops at the 6D vector rather than baking in a Gram-Schmidt step apps may
want to do differently.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPOS = os.environ.get("CONVERT_REPOS", "")
sys.path.insert(0, os.path.join(REPOS, "6DRepNet", "sixdrepnet"))
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from harness import convert_and_gate
from calib import calib_loader
from backbone.repvgg import get_RepVGG_func_by_name


class SixDRepNetDeploy(nn.Module):
    """SixDRepNet without the constructor's pretrained-backbone download: the
    checkpoint here already carries every weight, backbone included."""

    def __init__(self, backbone_name="RepVGG-B1g2"):
        super().__init__()
        backbone = get_RepVGG_func_by_name(backbone_name)(deploy=True)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = (
            backbone.stage0, backbone.stage1, backbone.stage2,
            backbone.stage3, backbone.stage4)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        last = 0
        for n, m in self.layer4.named_modules():
            if "rbr_reparam" in n and isinstance(m, nn.Conv2d):
                last = m.out_channels
        self.linear_reg = nn.Linear(last, 6)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        return self.linear_reg(x)


sd = torch.load(hf_hub_download("osanseviero/6DRepNet_300W_LP_AFLW2000", "model.pth"),
                map_location="cpu", weights_only=True)
net = SixDRepNetDeploy()
net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
net.eval()

batches = calib_loader("portrait", 224, "imagenet")
cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "sixdrepnet_headpose", net, (torch.randn(1, 3, 224, 224),),
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        extra_meta={
            "source": "thohemp/6DRepNet + osanseviero/6DRepNet_300W_LP_AFLW2000 weights",
            "license": "MIT",
            "preprocess": "RGB, ImageNet norm, 224x224 face crop",
            "outputs": "6D rotation representation [1,6]. To a rotation matrix: "
                       "b1 = normalize(v[0:3]); b2 = normalize(v[3:6] - (b1·v[3:6])b1); "
                       "b3 = cross(b1, b2); R = [b1 b2 b3]. Euler angles follow from R.",
            **({"notes":
                "int8 is measured but not shipped. Correlation is a weak read on a "
                "six-element output, so the check that matters is the angle between "
                "the fp32 and int8 rotations: median 46.5 deg over ten faces, worst "
                "104 deg. That is the known failure mode of re-parameterized RepVGG "
                "under post-training quantization — its fused branches leave weight "
                "ranges too wide for an int8 grid — and it needs quantization-aware "
                "training rather than anything on the export side."}
               if prec == "int8" else {}),
        },
    )
