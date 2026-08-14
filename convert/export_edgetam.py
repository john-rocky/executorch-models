"""EdgeTAM (on-device SAM 2, CVPR 2025) -> 2x ExecuTorch XNNPACK .pte.

Same encoder/decoder split and prompt contract as export_sam21_tiny.py — an app
written against SAM2.1 here swaps the two .pte files and nothing else. The GPU
patches in the LiteRT build of this model (SE mean split, ZeroStuffConvT) are ML
Drift workarounds and are not needed on XNNPACK; only the IO design is reused.

  encoder: pixel (1,3,1024,1024) -> image_embed (1,256,64,64),
           feat_s0 (1,32,256,256), feat_s1 (1,64,128,128)
  decoder: (image_embed, feat_s0, feat_s1, points (1,1,1,2) px in 1024-space,
           labels (1,1,1) int) -> masks (1,1,3,256,256) logits, iou (1,1,3)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from harness import convert_and_gate
from transformers import EdgeTamModel

# Identity repeat_interleave(1, dim) confuses the delegate's runtime shape
# propagation once lowered — see export_sam21_tiny.py.
_orig_repeat_interleave = torch.Tensor.repeat_interleave


def _repeat_interleave_skip1(self, repeats, dim=None, **kwargs):
    if isinstance(repeats, int) and repeats == 1 and dim is not None:
        return self
    return _orig_repeat_interleave(self, repeats, dim=dim, **kwargs)


torch.Tensor.repeat_interleave = _repeat_interleave_skip1

# facebook/EdgeTAM ships only the original edgetam.pt; this is the same weights in
# transformers format (Apache-2.0), which is what EdgeTamModel can load.
CKPT = "yonigozlan/EdgeTAM-hf"
model = EdgeTamModel.from_pretrained(CKPT).eval()


class Encoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        feats = self.m.get_image_embeddings(x)  # [feat_s0, feat_s1, image_embed]
        # .contiguous(): channels_last at the .pte boundary poisons consumers.
        return feats[-1].contiguous(), feats[0].contiguous(), feats[1].contiguous()


class Decoder(nn.Module):
    def __init__(self, m, image_pos, dense):
        super().__init__()
        self.prompt_encoder = m.prompt_encoder
        self.mask_decoder = m.mask_decoder
        self.register_buffer("image_pos", image_pos)
        self.register_buffer("dense", dense)

    def forward(self, image_embed, feat_s0, feat_s1, points, labels):
        sparse, _ = self.prompt_encoder(
            input_points=points, input_labels=labels,
            input_boxes=None, input_masks=None)
        masks, iou, _, _ = self.mask_decoder(
            image_embeddings=image_embed,
            image_positional_embeddings=self.image_pos,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=self.dense,
            multimask_output=True,
            high_resolution_features=[feat_s0, feat_s1],
        )
        return masks, iou


pixel_values = torch.randn(1, 3, 1024, 1024)
points = torch.tensor([[[[512.0, 512.0]]]])
labels = torch.tensor([[[1]]])

captured = {}
hook = model.mask_decoder.register_forward_pre_hook(
    lambda mod, args, kwargs: captured.update(kwargs), with_kwargs=True)
with torch.no_grad():
    ref = model(pixel_values=pixel_values, input_points=points,
                input_labels=labels, multimask_output=True)
hook.remove()
image_pos = captured["image_positional_embeddings"].detach().clone()
dense = captured["dense_prompt_embeddings"].detach().clone()

enc = Encoder(model)
dec = Decoder(model, image_pos, dense)

with torch.no_grad():
    ie, s0, s1 = (t.contiguous() for t in enc(pixel_values))
    masks, iou = dec(ie, s0, s1, points, labels)
comp = (masks.reshape(-1) - ref.pred_masks.reshape(-1)).abs().max().item()
print(f"composition max_abs_diff vs full model: {comp:.3e}")
assert comp < 1e-4, "wrapper composition diverges from EdgeTamModel forward"

precisions = sys.argv[1:] or ["fp32"]
for prec in precisions:
    convert_and_gate(
        "edgetam_encoder", enc, (pixel_values,), runs=5, precision=prec,
        extra_meta={
            "source": CKPT,
            "license": "Apache-2.0",
            "preprocess": "RGB/255, imagenet norm (mean .485/.456/.406, std .229/.224/.225), 1024x1024",
            "outputs": "image_embed [1,256,64,64], feat_s0 [1,32,256,256], feat_s1 [1,64,128,128]",
        },
    )
    convert_and_gate(
        "edgetam_decoder", dec, (ie, s0, s1, points, labels), runs=10, precision=prec,
        extra_meta={
            "source": CKPT,
            "license": "Apache-2.0",
            "preprocess": "points: pixel coords in 1024x1024 space [1,1,N,2] fp32; "
                          "labels [1,1,N] int64 (1=fg, 0=bg); prompt encoder embedded",
            "outputs": "mask logits [1,1,3,256,256] (upsample 4x to 1024, >0 = fg), iou [1,1,3]",
        },
    )
