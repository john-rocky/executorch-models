# ExecuTorch-Models

Converted **.pte model zoo** for iOS / Android / edge with [ExecuTorch](https://pytorch.org/executorch/) XNNPACK.

Every model ships on Hugging Face with a verification card: parity vs PyTorch fp32
(max_abs_diff + correlation), measured latency, exact pre/post-processing spec, and license.
Conversion scripts for all models are in [`convert/`](convert/) — one common harness,
one small script per model.

Models ship in every precision that earns its slot. A reduced-precision `.pte` is
published only if it holds its accuracy against fp32 eager **and** comes out at
least 5% smaller; all variants take and return fp32 tensors, so switching precision
is a file swap. Where a precision did not make it, the card says so and why.

**If you like this repository, please give it a star.**

## Models

### Vision

Sizes are MB; `corr` is the worst per-output correlation against torch fp32 eager,
measured on a real image.

| Model | Task | fp32 | fp16 | int8 | License |
|-------|------|------|------|------|---------|
| [EdgeTAM](https://huggingface.co/mlboydaisuke/EdgeTAM-ExecuTorch) | promptable segmentation | **19.7 enc + 24.7 dec** | 12.6 dec | — | Apache-2.0 |
| [MobileSAM](https://huggingface.co/mlboydaisuke/MobileSAM-ExecuTorch) | promptable segmentation | 28.3 enc + 20.5 dec | 10.5 dec | **14.0 enc** | Apache-2.0 / MIT |
| [SAM2.1-hiera-tiny](https://huggingface.co/mlboydaisuke/SAM2.1-hiera-tiny-ExecuTorch) | promptable segmentation | 109 enc + 25 dec (E2E mask IoU 1.0000) | 55.6 enc + 12.6 dec | — | Apache-2.0 |
| [RT-DETRv2-S](https://huggingface.co/mlboydaisuke/RT-DETRv2-S-ExecuTorch) | object detection (no NMS) | 81 | — | — | Apache-2.0 |
| [D-FINE-S](https://huggingface.co/mlboydaisuke/D-FINE-S-ExecuTorch) | object detection (no NMS) | 42 | — | — | Apache-2.0 |
| [YOLOX-s](https://huggingface.co/mlboydaisuke/YOLOX-s-ExecuTorch) | object detection | 36 | — | **9.2** (0.9988) | Apache-2.0 |
| [SSDLite320-MobileNetV3](https://huggingface.co/mlboydaisuke/SSDLite320-MobileNetV3-ExecuTorch) | object detection (raw head) | 14 | — | **3.9** (0.9636) | BSD-3 |
| [Depth-Anything-V2-Small](https://huggingface.co/mlboydaisuke/Depth-Anything-V2-Small-ExecuTorch) | monocular depth | 99 | 55.5 (1.0000) | **35.5** (1.0000) | Apache-2.0 |
| [DINOv2 ViT-S/14](https://huggingface.co/mlboydaisuke/DINOv2-ViT-S14-ExecuTorch) | feature extraction | 88 | 44.8 (0.9999) | **24.9** (0.9980) | Apache-2.0 |
| [CLIP ViT-B/32](https://huggingface.co/mlboydaisuke/CLIP-ViT-B32-ExecuTorch) | zero-shot classification | 352 img + 254 txt | 181 + 127 (1.0000) | **95.9** img (0.9957) | MIT |
| [MODNet](https://huggingface.co/mlboydaisuke/MODNet-ExecuTorch) | portrait matting | 26 | 24.4 (1.0000) | **6.8** (0.9999) | Apache-2.0 |
| [ormbg (ISNet)](https://huggingface.co/mlboydaisuke/ormbg-ExecuTorch) | background removal | 176 | — | **44.3** (1.0000) | Apache-2.0 |
| [DIS (IS-Net)](https://huggingface.co/mlboydaisuke/DIS-ISNet-ExecuTorch) | high-accuracy cutout | 176 | — | **44.3** (0.9878) | Apache-2.0 |
| [U²-Net](https://huggingface.co/mlboydaisuke/U2Net-ExecuTorch) | salient object segmentation | 176 | — | **44.3** (0.9802) | Apache-2.0 |
| [PIDNet-S](https://huggingface.co/mlboydaisuke/PIDNet-S-Cityscapes-ExecuTorch) | semantic segmentation | 31 | — | **7.9** (0.9989) | MIT |
| [TwinLiteNet](https://huggingface.co/mlboydaisuke/TwinLiteNet-ExecuTorch) | drivable area + lanes | 1.8 | — | — | MIT |
| [EDSR ×4](https://huggingface.co/mlboydaisuke/EDSR-x4-ExecuTorch) | super-resolution | 6.1 | — | **1.6** (0.9999) | Apache-2.0 |
| [Real-ESRGAN x4v3](https://huggingface.co/mlboydaisuke/Real-ESRGAN-x4v3-ExecuTorch) | super-resolution | **4.9** | — | — | BSD-3 |
| [EfficientNet-B1](https://huggingface.co/mlboydaisuke/EfficientNet-B1-ExecuTorch) | classification | 31 | 28.8 (0.9998) | — | BSD-3 |
| [6DRepNet](https://huggingface.co/mlboydaisuke/6DRepNet-HeadPose-ExecuTorch) | head pose (6D rotation) | 157 | — | — | MIT |

Every fp32 variant is corr 1.000000. A dash means that precision is not published —
each model card explains the specific reason.

### LLM (8da4w + 8-bit embedding, iPhone 17 Pro measured)

| Model | .pte | Decode (on-device) | License |
|-------|------|--------------------|---------|
| [LFM2.5-350M](https://huggingface.co/mlboydaisuke/LFM2.5-350M-ExecuTorch) | 253 MB | ~180 tok/s | LFM Open License |
| [LFM2.5-1.2B-Instruct](https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-Instruct-ExecuTorch) | 741 MB | 55–81 tok/s | LFM Open License |
| [Qwen3.5-0.8B](https://huggingface.co/mlboydaisuke/Qwen3.5-0.8B-ExecuTorch) | 651 MB | ~10 tok/s | Apache-2.0 |

All numbers are single-runtime ExecuTorch XNNPACK (CPU) measurements; details and
conditions are on each model card. LLM exports use `export_llm` configs in
[`llm_params/`](llm_params/) — instruct models **require the chat template**
(raw prompts return EOS immediately; see cards).

## Conversion recipe

```
torch.export → to_edge_transform_and_lower(XnnpackPartitioner) → .pte
```

[`convert/harness.py`](convert/harness.py) does export → lower → **parity gate vs
torch fp32 eager** (per-output max_abs_diff + correlation) → median latency →
delegate-coverage report → `results/*.json`, and
[`convert/gen_cards.py`](convert/gen_cards.py) renders the cards. Per-model scripts
are thin wrappers that:

- unwrap dict/dataclass outputs to tensors (harness accepts tensors/tuples only)
- tap raw detection heads instead of NMS-bearing forward paths
- fix layout at graph boundaries (`.contiguous()` — see SAM2.1 notes below)

## Picking a precision

Measured across every vision model above, not guessed:

**fp16 only pays off on transformers.** XNNPACK serializes convolution weights as
fp32 whatever dtype the graph carries (`op_conv2d.py` passes `force_fp32=True`), so
only weights that reach the delegate through `linear` actually shrink. ViT-based
models come out at 50–56% of fp32 (DINOv2 88.4 → 44.8 MB, CLIP image 351.6 → 180.7,
Depth-Anything-V2 99.0 → 55.5). Conv-only models come out at exactly 100% — ormbg,
PIDNet, TwinLiteNet, EDSR and SSDLite are byte-for-byte the fp32 file, plus cast
operations that make them slower. On a CNN, skip fp16 and go to int8.

**int8 is static for CNNs and dynamic for ViTs.** Static PT2E quantization takes
ViT outputs apart — DINOv2 drops to corr 0.380, CLIP image to 0.836,
Depth-Anything-V2 to 0.493. Quantizing only `linear` dynamically brings the same
models back to 0.998, 0.996 and 0.99998. Note that a *global* dynamic config also
annotates convolutions, which XNNPACK cannot lower (`ChannelsLastTaggedReshapePass:
required rank 4 tensor`), so scope it with `set_operator_type`.

**Calibrate and measure on real images.** A calibrated int8 model clips activations
it never saw, so parity measured on `torch.randn` reads far worse than the model
actually is. `convert/calib.py` loads a small image set for calibration and for the
parity gate.

**When fp16 breaks, check the model in eager before blaming the conversion.**
MobileSAM's TinyViT encoder returns corr -0.37 from a plain `model.half()` in eager,
with no ExecuTorch involved — some architectures simply are not half-precision safe,
and no amount of export-side work recovers that. Dynamic int8 was the answer for
that encoder instead (28.3 → 14.0 MB at corr 0.9999).

**Judge quality with the task's own metric when the output is small.** Correlation works on a 260k-element mask and says very little about a six-element regression. 6DRepNet's int8 build reads corr 0.815, which looks borderline; converting both outputs to rotation matrices puts it at a median 46° apart from fp32. Re-parameterized RepVGG backbones are a known post-training-quantization failure — the fused branches leave weight ranges too wide for an int8 grid.

**Some models just do not quantize.** EfficientNet-B1 lands at corr 0.077 whether
weights are per-channel or per-tensor, and whether the whole graph or only
conv/linear is annotated — with 100% of its ops on the delegate, this is a
quantization-accuracy limit, not a lowering bug. RT-DETRv2 and D-FINE break in fp16
(corr 0.33 / 0.22) because their decoders refine boxes as
`sigmoid(inverse_sigmoid(ref) + delta)`, which fp16 cannot resolve.

### Hard-won notes (traps)

- **channels_last poisoning (SAM2.1)**: transformers' `get_image_embeddings()`
  returns channels_last tensors; torch.export specializes the graph to that layout
  and XNNPACK's runtime shape propagation misreads it (phantom `(64,256,64,64)`
  resizes). Fix: `.contiguous()` on encoder outputs / decoder inputs.
- **Non-contiguous runtime inputs are silently misread**: numpy
  `transpose(...).astype(...)` keeps strides (`order='K'`); feeding the result via
  `torch.from_numpy` to the ExecuTorch runtime gives garbage without any error.
  Always `np.ascontiguousarray` first.
- **Graph asserts**: RT-DETRv2 / D-FINE carry `_is_all_true` runtime asserts that
  the Edge verifier rejects — `harness.py convert_and_gate(..., strip_asserts=True)`
  erases them before lowering.
- **XNNPACK PReLU segfault (executorch 1.4.0)**: PReLU constant data is not packed
  → use-after-free ([#17559](https://github.com/pytorch/executorch/issues/17559),
  fixed in [#21480](https://github.com/pytorch/executorch/pull/21480), not in 1.4.0).
  Workaround: exclude `PreluConfig` from the partitioner (see `convert/export_twinlite.py`).
- **Instruct LLMs need the chat template in benches too** — raw text returns
  `<|im_end|>` immediately and looks like broken generation.
- **Reshaping a constant inside the graph can silently corrupt its consumer**:
  MobileSAM's `TwoWayTransformer` opens with `image_pe.flatten(2).permute(0,2,1)`
  on a tensor that is constant for a fixed image size. Left in the graph, layer 0's
  keys come out at corr 0.78 against eager — with **no delegate involved**, and with
  every operator (attention, LayerNorm, MLP, flatten+permute) exact when exported on
  its own. Precompute the reshaped constant and pass it in as a buffer.
- **Quantized `slice` can stop a delegate from loading**: if PT2E gives a slice
  output a scale different from its input's, `xnn_define_static_slice` returns
  `xnn_status_invalid_parameter` and the whole method fails to load
  (`Init failed for backend XnnpackBackend`) — there is no partial fallback.
  It happens when a consumer such as `add` annotates the slice before
  `propagate_annotation` can share the producer's qspec. Workaround: drop
  `SliceCopyConfig` from the partitioner, or annotate conv/linear only.
- **fp16 needs data-dependent norms kept in fp32**: InstanceNorm / GroupNorm /
  LayerNorm compute variance from live activations, which overflows in fp16 and
  returns all-NaN (seen on MODNet's IBNorm). The harness keeps those layers in
  fp32 with cast boundaries; eval-mode BatchNorm is pure affine and stays fp16.

## Environment

- executorch 1.4.0 (pip), torch 2.13, Python 3.12
- LLM params JSONs are not bundled in the pip wheel — copy from the
  [executorch repo](https://github.com/pytorch/executorch) `examples/models/<family>/config/`
  at the matching tag.
