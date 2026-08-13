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


vis = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
x = torch.randn(1, 3, 224, 224)
convert_and_gate(
    "clip_vit_b32_image",
    ImageTower(vis),
    (x,),
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
convert_and_gate(
    "clip_vit_b32_text",
    TextTower(txt),
    (ids, mask),
    extra_meta={
        "source": "openai/clip-vit-base-patch32",
        "license": "MIT",
        "preprocess": "CLIP BPE tokens, fixed len 77 with attention mask",
        "outputs": "text embedding [1,512] (unnormalized; L2-normalize before cosine)",
    },
)
