# SAM 2.1 hiera-tiny — ExecuTorch XNNPACK (encoder + decoder)

Two .pte files (image-segment path: run the encoder once per image, the decoder
per click):

- `sam21_tiny_encoder_xnnpack_fp32.pte` (109.2 MB) — image (1,3,1024,1024) →
  image_embed (1,256,64,64), feat_s0 (1,32,256,256), feat_s1 (1,64,128,128)
- `sam21_tiny_decoder_xnnpack_fp32.pte` (24.7 MB) — (image_embed, feat_s0, feat_s1,
  points (1,1,N,2) fp32 pixel coords in 1024-space, labels (1,1,N) int64 1=fg/0=bg)
  → mask logits (1,1,3,256,256), iou scores (1,1,3)

- **Source**: facebook/sam2.1-hiera-tiny (transformers Sam2Model)
- **License**: Apache-2.0
- **Preprocess**: RGB/255, imagenet norm (mean .485/.456/.406, std .229/.224/.225),
  resize 1024x1024
- **Postprocess**: pick argmax(iou) of the 3 mask logits, threshold > 0, upsample
  4x (256→1024) to image space. The prompt encoder is embedded in the decoder —
  pass raw click coordinates, no separate point-encoding code needed.

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager (random input): all 5 outputs corr 1.000000
(worst max_abs_diff 2.2e-05). End-to-end two-.pte chain on a synthetic image:
mask IoU vs torch = 1.0000, iou_scores match to 6 decimals.

Median latency (Mac, reference only — ViT-heavy graphs are known to time poorly
on Mac XNNPACK; device numbers to follow): encoder 2506 ms, decoder 20 ms.

## Conversion notes

torch.export → to_edge_transform_and_lower(XnnpackPartitioner) → .pte, with:
- `.contiguous()` on the encoder outputs / decoder inputs (transformers emits
  channels_last; leaving it poisons XNNPACK's runtime shape propagation)
- identity `repeat_interleave(1, dim)` calls stripped (delegate mis-sizes their
  decomposition)
- constant `dense_prompt_embeddings` (no-mask embedding) baked as a buffer
