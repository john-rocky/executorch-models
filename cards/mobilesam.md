# MobileSAM — ExecuTorch XNNPACK (encoder + decoder)

Promptable segmentation in two `.pte` files: run the encoder once per image, the
decoder once per click.

- `mobilesam_encoder_xnnpack_fp32.pte` (28.3 MB) — image (1,3,1024,1024) →
  image_embed (1,256,64,64)
- `mobilesam_decoder_xnnpack_fp32.pte` (20.5 MB) — (image_embed,
  points (1,N,2) fp32 pixel coords in 1024-space, labels (1,N) fp32 1=fg/0=bg)
  → mask logits (1,3,256,256), iou scores (1,3)

MobileSAM is SAM with its ViT-H encoder replaced by TinyViT. Same prompt contract as
the [SAM2.1](https://huggingface.co/mlboydaisuke/SAM2.1-hiera-tiny-ExecuTorch) and
[EdgeTAM](https://huggingface.co/mlboydaisuke/EdgeTAM-ExecuTorch) conversions, with
two differences worth knowing: this decoder needs only the image embedding (no
high-resolution feature maps), and labels are fp32 rather than int64.

- **Source**: [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM),
  weights from [dhkim2810/MobileSAM](https://huggingface.co/dhkim2810/MobileSAM)
- **License**: Apache-2.0 (code) / MIT (weights)
- **Preprocess**: RGB, SAM norm (mean 123.675/116.28/103.53, std 58.395/57.12/57.375),
  resize the longest side to 1024 and pad to 1024×1024
- **Postprocess**: take argmax(iou) of the 3 mask logits, threshold at > 0, upsample
  4× (256→1024) back to image space, then crop the padding. The prompt encoder is
  inside the decoder — pass raw click coordinates.

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Both graphs match torch fp32 eager at corr 1.000000, and the wrappers compose back
to the stock modules exactly (max_abs_diff 0.000e+00).

| graph | output | shape | max_abs_diff | corr |
|-------|--------|-------|--------------|------|
| encoder | image_embed | [1, 256, 64, 64] | 3.815e-06 | 1.000000 |
| decoder | mask logits | [1, 3, 256, 256] | 1.717e-05 | 1.000000 |
| decoder | iou | [1, 3] | 1.192e-07 | 1.000000 |

Median over 10 runs, Mac arm64 single process — a relative reference, not a device
number: encoder 130.3 ms (torch eager 138.7 ms), decoder 20.5 ms (eager 11.2 ms).
XNNPACK delegate coverage: encoder 89.0%, decoder 80.6%.

## Conversion

torch.export → to_edge_transform_and_lower(XnnpackPartitioner) → .pte
(conversion script: [executorch-models](https://github.com/john-rocky/executorch-models))

Three rewrites were needed, and the reasons generalize to other SAM-family ports:

**The constant positional embedding is precomputed.** `TwoWayTransformer.forward`
opens with `image_pe.flatten(2).permute(0, 2, 1)`. That input is constant for a
fixed image size, and leaving the reshape in the graph corrupts the block that
consumes it — layer 0's keys came out at corr 0.78 against eager. Handing the
transformer the already-flat tensor restores corr 1.000000. This reproduces with no
delegate at all, and every operator involved verifies clean in isolation, so it is
worth knowing about rather than rediscovering.

**Boolean-mask assignment is rewritten as arithmetic.** The prompt encoder writes
`point_embedding[labels == -1] = 0.0` and three more masked `+=`. torch.export turns
each into an `index_put` behind a `nonzero`, which is a data-dependent shape. The
equivalent `emb * (1 - m) + m * w` form has fixed shapes and no runtime guards.

**Identity `repeat_interleave` is dropped.** The mask decoder calls
`torch.repeat_interleave(x, tokens.shape[0], dim=0)` with one point batch — a no-op
whose lowered form the delegate mis-sizes. Note this model uses the functional form,
not the tensor method.
