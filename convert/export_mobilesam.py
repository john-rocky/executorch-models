"""MobileSAM (SAM v1 with a TinyViT encoder) -> 2x ExecuTorch XNNPACK .pte.

Same two-file split as export_sam21_tiny.py / export_edgetam.py: run the encoder
once per image, the decoder per click. The LiteRT build of this model replaces
GELU with a sigmoid approximation because TFLite has no Erf; XNNPACK has GELU, so
the stock model exports unchanged. Only the weight source and the IO split are
reused from that script.

  encoder: pixel (1,3,1024,1024) -> image_embed (1,256,64,64)
  decoder: (image_embed, points (1,N,2) px in 1024-space, labels (1,N))
           -> masks (1,3,256,256) logits, iou (1,3)

`multimask_output=True` returns 3 masks here, the same count as SAM 2.1: SAM v1's
decoder predicts 4 and slices the first one off. Verified against the exported
graph rather than assumed — see convert/verify_cards.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPOS = os.environ.get("CONVERT_REPOS", "")
sys.path.insert(0, os.path.join(REPOS, "MobileSAM"))
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from harness import convert_and_gate
from mobile_sam import sam_model_registry

# The mask decoder does `torch.repeat_interleave(x, tokens.shape[0], dim=0)` with
# one point batch, i.e. a no-op that the delegate's runtime shape propagation then
# mis-sizes. Drop identity repeats. Unlike SAM 2.1 this model calls the functional
# form, so patch that as well as the Tensor method.
_orig_fn = torch.repeat_interleave
_orig_method = torch.Tensor.repeat_interleave


def _skip1_fn(input, repeats, dim=None, **kw):
    if isinstance(repeats, int) and repeats == 1 and dim is not None:
        return input
    return _orig_fn(input, repeats, dim=dim, **kw)


def _skip1_method(self, repeats, dim=None, **kw):
    if isinstance(repeats, int) and repeats == 1 and dim is not None:
        return self
    return _orig_method(self, repeats, dim=dim, **kw)


torch.repeat_interleave = _skip1_fn
torch.Tensor.repeat_interleave = _skip1_method

ckpt = hf_hub_download("dhkim2810/MobileSAM", "mobile_sam.pt")
sam = sam_model_registry["vit_t"](checkpoint=ckpt).eval()


class Encoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.enc = m.image_encoder

    def forward(self, x):
        # .contiguous() at the .pte boundary: see export_sam21_tiny.py.
        return self.enc(x).contiguous()


class Decoder(nn.Module):
    """Embeds the prompt encoder and bakes the constant dense positional
    embedding as a buffer, so apps pass raw click coordinates.

    The stock `_embed_points` writes through boolean masks
    (`point_embedding[labels == -1] = 0.0`, then three `+=` of the same shape).
    torch.export turns each of those into an `index_put` behind a `nonzero`,
    which is a data-dependent shape: the graph carries `_assert_scalar` guards,
    the ops land on the portable kernels, and the mask logits come back at corr
    0.76 against eager. Rewritten below as select-and-add arithmetic — same
    result, fixed shapes, no guards."""

    def __init__(self, m, image_pe):
        super().__init__()
        self.pe_layer = m.prompt_encoder.pe_layer
        self.mask_decoder = m.mask_decoder
        self.input_image_size = m.prompt_encoder.input_image_size
        self.image_embedding_size = m.prompt_encoder.image_embedding_size
        self.register_buffer("image_pe", image_pe)
        self.register_buffer("w_not_a_point", m.prompt_encoder.not_a_point_embed.weight)
        self.register_buffer("w_bg", m.prompt_encoder.point_embeddings[0].weight)
        self.register_buffer("w_fg", m.prompt_encoder.point_embeddings[1].weight)
        self.register_buffer("w_no_mask", m.prompt_encoder.no_mask_embed.weight)

    def embed_points(self, points, labels):
        # Stock forward pads one dummy point with label -1 when no box is given.
        pad_pt = torch.zeros((points.shape[0], 1, 2), dtype=points.dtype)
        pad_lbl = -torch.ones((labels.shape[0], 1), dtype=labels.dtype)
        points = torch.cat([points + 0.5, pad_pt], dim=1)
        labels = torch.cat([labels, pad_lbl], dim=1)

        coords = torch.stack([points[..., 0] / self.input_image_size[1],
                              points[..., 1] / self.input_image_size[0]], dim=-1)
        emb = self.pe_layer._pe_encoding(coords)

        # labels == -1 zeroes the positional part before adding not_a_point.
        m_pad = (labels == -1).unsqueeze(-1).to(emb.dtype)
        m_bg = (labels == 0).unsqueeze(-1).to(emb.dtype)
        m_fg = (labels == 1).unsqueeze(-1).to(emb.dtype)
        return (emb * (1 - m_pad)
                + m_pad * self.w_not_a_point
                + m_bg * self.w_bg
                + m_fg * self.w_fg)

    def forward(self, image_embed, points, labels):
        sparse = self.embed_points(points, labels)
        dense = self.w_no_mask.reshape(1, -1, 1, 1).expand(
            1, -1, self.image_embedding_size[0], self.image_embedding_size[1])
        masks, iou = self.mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        return masks, iou


def bake_transformer_pe(transformer, image_pe):
    """TwoWayTransformer.forward starts with
    `image_pe.flatten(2).permute(0, 2, 1)`. That input is a constant here, and
    computing the reshape in-graph on a constant silently corrupts the block that
    consumes it — the keys coming out of layer 0 land at corr 0.78 against eager,
    fully portable, with every op verified correct in isolation. Precomputing the
    same tensor and handing the transformer the already-flat version restores
    corr 1.000000. Same class as the SAM 2.1 constant-subgraph bake.

    Patching happens on the class and the tensor lives in a buffer, both
    deliberately. A closure over the module instance would survive
    `copy.deepcopy(model).half()` still pointing at the *original* fp32 module,
    so the fp16 export would meet an fp32 constant ("expected scalar type
    torch.float16 but found torch.float32"); going through `self` lets each copy
    read its own converted buffer.
    """
    transformer.register_buffer(
        "flat_pe", image_pe.flatten(2).permute(0, 2, 1).contiguous())
    cls = type(transformer)
    orig_forward = cls.forward

    def forward(self, image_embedding, image_pe_arg, point_embedding):
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        queries, keys = point_embedding, image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys,
                                  query_pe=point_embedding, key_pe=self.flat_pe)
        q = queries + point_embedding
        k = keys + self.flat_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = self.norm_final_attn(queries + attn_out)
        return queries, keys

    cls.forward = forward
    return orig_forward


pixel_values = torch.randn(1, 3, 1024, 1024)
points = torch.tensor([[[512.0, 512.0]]])
labels = torch.tensor([[1]], dtype=torch.float32)

with torch.no_grad():
    image_pe = sam.prompt_encoder.get_dense_pe().detach().clone()

enc = Encoder(sam)
dec = Decoder(sam, image_pe)
bake_transformer_pe(sam.mask_decoder.transformer, image_pe)

with torch.no_grad():
    ie = enc(pixel_values)
    masks, iou = dec(ie, points, labels)
    # Reference: the same path through the stock modules, no wrappers.
    ref_ie = sam.image_encoder(pixel_values)
    sp, dn = sam.prompt_encoder(points=(points, labels), boxes=None, masks=None)
    ref_masks, _ = sam.mask_decoder(
        image_embeddings=ref_ie, image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sp, dense_prompt_embeddings=dn,
        multimask_output=True)
comp = (masks - ref_masks).abs().max().item()
print(f"composition max_abs_diff vs stock modules: {comp:.3e}")
assert comp < 1e-4, "wrapper composition diverges from the stock model"

for prec in (sys.argv[1:] or ["fp32"]):
    # The encoder is skipped in fp16: TinyViT does not survive half precision,
    # and not because of anything ExecuTorch does — plain `model.half()` in eager
    # already returns corr -0.37 against the fp32 model, with or without keeping
    # the data-stat norms in fp32. The decoder is a plain transformer and halves
    # fine, so it is still built.
    if prec == "fp16":
        print("skip encoder fp16: TinyViT is not fp16-safe in eager (corr -0.37)")
    else:
        convert_and_gate(
            "mobilesam_encoder", enc, (pixel_values,), runs=5, precision=prec,
            int8_dynamic=True,  # TinyViT: static int8 wrecks attention
            extra_meta={
                "source": "ChaoningZhang/MobileSAM + dhkim2810/MobileSAM weights",
                "license": "Apache-2.0 (code) / MIT (weights)",
                "preprocess": "RGB, SAM norm (mean 123.675/116.28/103.53, std 58.395/57.12/57.375), "
                              "longest side 1024 then pad to 1024x1024",
                "outputs": "image_embed [1,256,64,64]",
            },
        )
    convert_and_gate(
        "mobilesam_decoder", dec, (ie, points, labels), runs=10, precision=prec,
        int8_dynamic=True,
        extra_meta={
            "source": "ChaoningZhang/MobileSAM + dhkim2810/MobileSAM weights",
            "license": "Apache-2.0 (code) / MIT (weights)",
            "preprocess": "points: pixel coords in the padded 1024x1024 space [1,N,2] fp32; "
                          "labels [1,N] fp32 (1=fg, 0=bg, -1=pad); prompt encoder embedded",
            "outputs": "mask logits [1,3,256,256] (upsample 4x to 1024, >0 = fg), iou [1,3]",
        },
    )
