"""CLIP ViT-B/32 -> 2x ExecuTorch XNNPACK .pte (image tower + text tower)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import convert_and_gate
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection


class ImageTower(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).image_embeds


class TextTower(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids, mask):
        return self.m(input_ids=ids, attention_mask=mask).text_embeds


from calib import calib_loader

vis = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
x = torch.randn(1, 3, 224, 224)
batches = calib_loader("general", 224, "clip")
cal = lambda mod: [mod(*b) for b in batches]
for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "clip_vit_b32_image",
        ImageTower(vis),
        (x,),
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        int8_dynamic=True,  # ViT: static int8 wrecks embeddings (corr 0.84); dynamic holds
        extra_meta={
            "source": "openai/clip-vit-base-patch32",
            "license": "MIT",
            "preprocess": "RGB, CLIP norm (mean .481/.458/.408, std .269/.261/.276), 224x224",
            "outputs": "image embedding [1,512] (unnormalized; L2-normalize before cosine)",
        },
    )

txt = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
ids = torch.zeros(1, 77, dtype=torch.long)
ids[0, 0] = 49406  # BOS
ids[0, 1] = 49407  # EOS
mask = torch.ones(1, 77, dtype=torch.long)
# text tower: no image calibration path — fp32 + fp16 only
for prec in (sys.argv[1:] or ["fp32", "fp16"]):
    if prec == "int8":
        continue
    convert_and_gate(
        "clip_vit_b32_text",
        TextTower(txt),
        (ids, mask),
        precision=prec,
        extra_meta={
            "source": "openai/clip-vit-base-patch32",
            "license": "MIT",
            "preprocess": "CLIP BPE tokens, fixed len 77 with attention mask",
            "outputs": "text embedding [1,512] (unnormalized; L2-normalize before cosine)",
        },
    )
