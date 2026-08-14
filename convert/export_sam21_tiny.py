"""SAM 2.1 hiera-tiny -> 2x ExecuTorch XNNPACK .pte (image encoder + prompt decoder).

Image-segment path only (encoder once + decoder per click), matching the LiteRT
split (LiteRT-Models/sam2). XNNPACK needs none of the GPU patches from that
script — stock transformers Sam2 exports as-is; only the IO split design and
the point-encoding spec are reused.

  encoder: pixel (1,3,1024,1024) -> image_embed (1,256,64,64),
           feat_s0 (1,32,256,256), feat_s1 (1,64,128,128)
  decoder: (image_embed, feat_s0, feat_s1, points (1,1,1,2) px in 1024-space,
           labels (1,1,1) int) -> masks (1,1,3,256,256) logits, iou (1,1,3)

The decoder embeds the prompt encoder and bakes the (constant, size-fixed)
image positional embeddings and no-mask dense embeddings as buffers, so apps
pass raw click coordinates — no separate point-encoder code needed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from harness import convert_and_gate
from transformers import Sam2Model

# The decoder calls repeat_interleave(point_batch_size, ...) everywhere; with a
# static single-point export these are identities, and the XNNPACK delegate's
# shape propagation mis-sizes their decomposition at runtime (output resized to
# (64,256,64,64) where (1,256,64,64) is expected). Drop identity repeats.
_orig_repeat_interleave = torch.Tensor.repeat_interleave


def _repeat_interleave_skip1(self, repeats, dim=None, **kwargs):
    if isinstance(repeats, int) and repeats == 1 and dim is not None:
        return self
    return _orig_repeat_interleave(self, repeats, dim=dim, **kwargs)


torch.Tensor.repeat_interleave = _repeat_interleave_skip1

CKPT = "facebook/sam2.1-hiera-tiny"
model = Sam2Model.from_pretrained(CKPT).eval()


class Encoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        feats = self.m.get_image_embeddings(x)  # [feat_s0, feat_s1, image_embed]
        # .contiguous(): the encoder emits channels_last tensors in eager; leaving
        # that layout at the .pte boundary poisons every consumer (portable kernels
        # reject mixed dim orders, and the XNNPACK delegate's shape propagation
        # reads the physical layout as logical dims -> phantom (64,256,64,64)).
        return feats[-1].contiguous(), feats[0].contiguous(), feats[1].contiguous()


class Decoder(nn.Module):
    """image/prompt -> masks. dense_prompt_embeddings (the no-mask embedding
    expand) is constant for this export and baked as a buffer — leaving its
    expand subgraph in the graph makes the XNNPACK delegate's runtime shape
    propagation miscompute the following add's output as (64,256,64,64)."""

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

# Capture the constant image positional embeddings from one full forward.
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

# Composition check (pure torch): wrappers must reproduce the full model.
# Inputs to the decoder export MUST be contiguous — see Encoder.forward note.
with torch.no_grad():
    ie, s0, s1 = (t.contiguous() for t in enc(pixel_values))
    masks, iou = dec(ie, s0, s1, points, labels)
comp = (masks.reshape(-1) - ref.pred_masks.reshape(-1)).abs().max().item()
print(f"composition max_abs_diff vs full model: {comp:.3e}")
assert comp < 1e-4, "wrapper composition diverges from Sam2Model forward"

# PRECISIONS env var, not argv: this script already parses --dec-only/--exclude/etc.
PRECISIONS = os.environ.get("PRECISIONS", "fp32").split(",")

for prec in PRECISIONS:
    if "--dec-only" in sys.argv:
        break
    convert_and_gate(
        "sam21_tiny_encoder", enc, (pixel_values,), runs=5, precision=prec,
        int8_dynamic=True,  # Hiera is attention-heavy: static int8 wrecks it
        extra_meta={
            "source": CKPT,
            "license": "Apache-2.0",
            "preprocess": "RGB/255, imagenet norm (mean .485/.456/.406, std .229/.224/.225), 1024x1024",
            "outputs": "image_embed [1,256,64,64], feat_s0 [1,32,256,256], feat_s1 [1,64,128,128]",
        },
    )

exclude = set()
for i, a in enumerate(sys.argv):
    if a == "--exclude":
        exclude = set(sys.argv[i + 1].split(","))
if exclude or "--per-op" in sys.argv:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.partition.config import ALL_PARTITIONER_CONFIGS
    configs = [c for c in ALL_PARTITIONER_CONFIGS if c.__name__ not in exclude]
    dec_partitioner = XnnpackPartitioner(configs=configs, per_op_mode="--per-op" in sys.argv,
                                         verbose="--verbose" in sys.argv)
elif "--portable" in sys.argv:
    dec_partitioner = False
else:
    dec_partitioner = None

for prec in PRECISIONS:
    convert_and_gate(
        "sam21_tiny_decoder", dec, (ie, s0, s1, points, labels), runs=10,
        partitioner=dec_partitioner,
        skip_dim_order="--skip-dim-order" in sys.argv,
        precision=prec,
        int8_dynamic=True,
        extra_meta={
            "source": CKPT,
            "license": "Apache-2.0",
            "preprocess": "points: pixel coords in 1024x1024 space [1,1,N,2] fp32; "
                          "labels [1,1,N] int64 (1=fg, 0=bg); prompt encoder embedded",
            "outputs": "mask logits [1,1,3,256,256] (upsample 4x to 1024, >0 = fg), iou [1,1,3]",
        },
    )
