# EdgeTAM — ExecuTorch XNNPACK (encoder + decoder)

Promptable segmentation in two `.pte` files: run the encoder once per image, the
decoder once per click.

- `edgetam_encoder_xnnpack_fp32.pte` (19.7 MB) — image (1,3,1024,1024) →
  image_embed (1,256,64,64), feat_s0 (1,32,256,256), feat_s1 (1,64,128,128)
- `edgetam_decoder_xnnpack_fp32.pte` (24.7 MB) — (image_embed, feat_s0, feat_s1,
  points (1,1,N,2) fp32 pixel coords in 1024-space, labels (1,1,N) int64 1=fg/0=bg)
  → mask logits (1,1,3,256,256), iou scores (1,1,3)
- `edgetam_decoder_xnnpack_fp16.pte` (12.6 MB) — the same decoder at half the size,
  corr 1.000000 against fp32 eager. It takes and returns fp32 tensors, so pairing it
  with the fp32 encoder needs no app changes.

The encoder ships in fp32 only, and that is not an omission. Its backbone is RepViT,
which is convolutional, and XNNPACK serializes convolution weights as fp32 whatever
dtype the graph carries — fp16 came out at 19.8 MB (100.5%) and dynamic int8 at
19.7 MB, so neither buys anything. At 19.7 MB the fp32 encoder is already smaller
than SAM 2.1 hiera-tiny's *fp16* encoder (55.6 MB).

EdgeTAM is Meta's on-device SAM 2 (CVPR 2025). Its encoder is **5.5× smaller than
SAM 2.1 hiera-tiny's** (19.7 MB vs 109.2 MB) for the same output contract, so an
app written against
[SAM2.1-hiera-tiny-ExecuTorch](https://huggingface.co/mlboydaisuke/SAM2.1-hiera-tiny-ExecuTorch)
swaps the two files and changes nothing else.

- **Source**: [facebook/EdgeTAM](https://huggingface.co/facebook/EdgeTAM), loaded
  from the transformers-format mirror
  [yonigozlan/EdgeTAM-hf](https://huggingface.co/yonigozlan/EdgeTAM-hf)
  (`facebook/EdgeTAM` publishes only the original `edgetam.pt`)
- **License**: Apache-2.0
- **Preprocess**: RGB/255, ImageNet norm (mean .485/.456/.406, std .229/.224/.225),
  resize 1024×1024
- **Postprocess**: take argmax(iou) of the 3 mask logits, threshold at > 0, upsample
  4× (256→1024) back to image space. The prompt encoder is inside the decoder — pass
  raw click coordinates, no separate point-encoding code needed.

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Every output of both graphs matches torch fp32 eager at corr 1.000000, and the two
wrappers compose back to `EdgeTamModel.forward` exactly (max_abs_diff 0.000e+00).

| graph | output | shape | max_abs_diff | corr |
|-------|--------|-------|--------------|------|
| encoder | image_embed | [1, 256, 64, 64] | 0.000e+00 | 1.000000 |
| encoder | feat_s0 | [1, 32, 256, 256] | 0.000e+00 | 1.000000 |
| encoder | feat_s1 | [1, 64, 128, 128] | 0.000e+00 | 1.000000 |
| decoder | mask logits | [1, 1, 3, 256, 256] | 0.000e+00 | 1.000000 |
| decoder | iou | [1, 1, 3] | 0.000e+00 | 1.000000 |

Median over 10 runs, Mac arm64 single process — a relative reference, not a device
number: encoder 32.3 ms (torch eager 103.5 ms), decoder 23.7 ms (eager 13.9 ms).

XNNPACK delegate coverage: encoder 99.8% (one `upsample_nearest2d` on the portable
kernels), decoder 66.5% (the prompt encoder's `expand`/`where` bookkeeping stays on
portable; every convolution and matmul is delegated).

## Conversion

torch.export → to_edge_transform_and_lower(XnnpackPartitioner) → .pte
(conversion script: [executorch-models](https://github.com/john-rocky/executorch-models))

Two details matter for this split, both shared with the SAM 2.1 conversion. Encoder
outputs are forced `.contiguous()` — transformers hands back channels_last tensors,
and that layout at a `.pte` boundary makes the delegate's runtime shape propagation
read physical strides as logical dims. Identity `repeat_interleave(1, dim)` calls in
the decoder are dropped, since their lowered form mis-sizes on a single-point export.

The GPU-specific rewrites in the LiteRT build of this model (splitting the
squeeze-excite mean, replacing ConvTranspose2d) are ML Drift workarounds and are not
needed here — XNNPACK runs the stock graph.
