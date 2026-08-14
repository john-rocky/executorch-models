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
| [YOLOX-s](https://huggingface.co/mlboydaisuke/YOLOX-s-ExecuTorch) | object detection | 36 | — | **9.2** (0.9998) | Apache-2.0 |
| [SSDLite320-MobileNetV3](https://huggingface.co/mlboydaisuke/SSDLite320-MobileNetV3-ExecuTorch) | object detection (raw head) | 14 | — | **3.9** (0.9688) | BSD-3 |
| [Depth-Anything-V2-Small](https://huggingface.co/mlboydaisuke/Depth-Anything-V2-Small-ExecuTorch) | monocular depth | 99 | 55.5 (1.0000) | **35.5** (1.0000) | Apache-2.0 |
| [MoGe-2 ViT-S](https://huggingface.co/mlboydaisuke/MoGe-2-ViT-S-ExecuTorch) | point map + normals + mask + scale | 141 | 96.6 | **76.4** | MIT |
| [DINOv2 ViT-S/14](https://huggingface.co/mlboydaisuke/DINOv2-ViT-S14-ExecuTorch) | feature extraction | 88 | 44.8 (0.9999) | **24.9** (0.9980) | Apache-2.0 |
| [CLIP ViT-B/32](https://huggingface.co/mlboydaisuke/CLIP-ViT-B32-ExecuTorch) | zero-shot classification | 352 img + 254 txt | 181 + 127 (1.0000) | **95.9** img (0.9957) | MIT |
| [MODNet](https://huggingface.co/mlboydaisuke/MODNet-ExecuTorch) | portrait matting | 26 | 24.4 (1.0000) | **6.8** (0.9999) | Apache-2.0 |
| [ormbg (ISNet)](https://huggingface.co/mlboydaisuke/ormbg-ExecuTorch) | background removal | 176 | — | **44.3** (1.0000) | Apache-2.0 |
| [DIS (IS-Net)](https://huggingface.co/mlboydaisuke/DIS-ISNet-ExecuTorch) | high-accuracy cutout | 176 | — | — | Apache-2.0 |
| [U²-Net](https://huggingface.co/mlboydaisuke/U2Net-ExecuTorch) | salient object segmentation | 176 | — | — | Apache-2.0 |
| [PIDNet-S](https://huggingface.co/mlboydaisuke/PIDNet-S-Cityscapes-ExecuTorch) | semantic segmentation | 31 | — | **7.9** (0.9989) | MIT |
| [TwinLiteNet](https://huggingface.co/mlboydaisuke/TwinLiteNet-ExecuTorch) | drivable area + lanes | 1.8 | — | — | MIT |
| [EDSR ×4](https://huggingface.co/mlboydaisuke/EDSR-x4-ExecuTorch) | super-resolution | 6.1 | — | **1.6** (0.9999) | Apache-2.0 |
| [Real-ESRGAN x4v3](https://huggingface.co/mlboydaisuke/Real-ESRGAN-x4v3-ExecuTorch) | super-resolution | **4.9** | — | — | BSD-3 |
| [LaMa](https://huggingface.co/mlboydaisuke/LaMa-Inpainting-ExecuTorch) | inpainting (512x512) | 205 | — | — | Apache-2.0 |
| [EfficientNet-B1](https://huggingface.co/mlboydaisuke/EfficientNet-B1-ExecuTorch) | classification | 31 | 28.8 (0.9998) | — | BSD-3 |
| [6DRepNet](https://huggingface.co/mlboydaisuke/6DRepNet-HeadPose-ExecuTorch) | head pose (6D rotation) | 157 | — | — | MIT |
| [RTMPose-s](https://huggingface.co/mlboydaisuke/RTMPose-s-Body-ExecuTorch) | 2D body pose (17 kpts) | **21.9** | — | — | Apache-2.0 |
| [RTMPose-m Hand](https://huggingface.co/mlboydaisuke/RTMPose-m-Hand-ExecuTorch) | hand pose (21 kpts) | 55.1 | — | — | Apache-2.0 |
| [RTMPose-m Face](https://huggingface.co/mlboydaisuke/RTMPose-m-Face-ExecuTorch) | face landmarks (106 kpts) | 67.9 | — | — | Apache-2.0 |
| [RTMPose-m Animal](https://huggingface.co/mlboydaisuke/RTMPose-m-Animal-ExecuTorch) | animal pose (17 kpts, AP-10K) | 54.5 | — | — | Apache-2.0 |

Every fp32 variant is corr 1.000000. A dash means that precision is not published —
each model card explains the specific reason.

Every published int8 build has also been checked in its own units, not just in
correlation — mask IoU for the segmenters, PSNR for image-to-image, post-NMS
detection agreement for the detectors, cosine similarity for the embedding models,
delta-1 for depth. Those numbers are on the cards and `convert/audit_int8.py`
produces them:

| model | measured | result |
|-------|----------|--------|
| ormbg / MODNet | mask IoU | 0.999 / 0.992 |
| DIS | mask IoU | 0.963 |
| PIDNet | pixels keeping their class | 95.3% |
| EDSR | PSNR | 47.4 dB |
| SSDLite | firing detections agreeing | 99.9% |
| YOLOX | post-NMS detections matched | 93.8% |
| DINOv2 / CLIP | embedding cosine | 0.997 / 0.996 |
| MobileSAM encoder | embedding cosine | 1.000 |
| Depth-Anything-V2 | delta-1 vs fp32 | 0.994 |
| Whisper encoder | hidden-state cosine | 0.9993 |
| MoGe-2 | point map / normals cosine | 0.999 |

One build did not survive and was withdrawn: U²-Net's int8 shrinks weak saliency
badly (mask IoU 0.21 at worst) where correlation read 0.98.

### Audio

| Model | Task | fp32 | fp16 | int8 | License |
|-------|------|------|------|------|---------|
| [Whisper-tiny](https://huggingface.co/mlboydaisuke/Whisper-tiny-ExecuTorch) | speech recognition | 32.9 enc + 198 dec | 17.6 enc + 99.1 dec | **11.7 enc** | Apache-2.0 |

Split encoder/decoder: the encoder runs once per 30-second window, the decoder once
per generated token. Static 128-token window, no KV cache — a greedy step is argmax,
append, re-run. The decoder `.pte` is larger than its 118 MB of weights because
Whisper's tied vocabulary matrix needs two representations in one file: an embedding
table to index and a delegate-packed copy for the output matmul.

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

**Calibrate on the input the model is designed to consume.** Both detectors here were first quantized against landscape photos containing almost no COCO objects, which moved YOLOX's boxes by up to 12 px; re-calibrating on street scenes took its correlation from 0.9988 to 0.9998. RTMPose is the sharper case — it is top-down, so it wants a crop around one person, and on whole photographs the fp32 model produces no coherent pose at all. `convert/make_person_crops.py` builds that set by running the shipped YOLOX over the same photos.

**Calibrate on imagery the model will actually meet.** Both detectors here were first quantized against landscape photos containing almost no COCO objects, which moved YOLOX's boxes by up to 12 px. Re-calibrating on street scenes — 653 firing detections instead of nearly none — took its correlation from 0.9988 to 0.9998 and its post-NMS agreement with fp32 to 94%.

**Calibrate and measure on real images.** A calibrated int8 model clips activations
it never saw, so parity measured on `torch.randn` reads far worse than the model
actually is. `convert/calib.py` loads a small image set for calibration and for the
parity gate.

**When fp16 breaks, check the model in eager before blaming the conversion.**
MobileSAM's TinyViT encoder returns corr -0.37 from a plain `model.half()` in eager,
with no ExecuTorch involved — some architectures simply are not half-precision safe,
and no amount of export-side work recovers that. Dynamic int8 was the answer for
that encoder instead (28.3 → 14.0 MB at corr 0.9999).

**Correlation is a first filter, not the verdict — finish with the task's own metric.** It misses in both directions. LaMa's int8 build clears the 0.95 bar at 0.958 and is 22 dB PSNR against fp32, which is visible degradation, so it is not published. EDSR's int8 reads 0.9999 and measures 47 dB, which really is invisible. 6DRepNet's reads 0.815 and works out to 46° of rotation error — a genuine failure, and the known one for re-parameterized RepVGG, whose fused branches leave weight ranges too wide for an int8 grid. Use PSNR for image-to-image, IoU for masks, degrees for pose, box/score agreement for detection.

**Some models just do not quantize.** EfficientNet-B1's int8 build keeps the fp32
top-1 label on 0 of 10 images, and the fp32 label is not even in its top-5. Neither
per-tensor weights, nor annotating only conv/linear, nor leaving the squeeze-excite
blocks in fp32 moves it, and 100% of its ops are on the delegate — this is a
quantization-accuracy limit that wants cross-layer equalisation or QAT, not
anything on the export side. Worth noting that locating the damage by correlating
intermediate features does not work either: cut at successive stages it reads
0.57, 0.39, 0.31, 0.17, 0.66 — non-monotonic, because correlation between post-SiLU
feature maps means nothing. RT-DETRv2 and D-FINE break in fp16
(corr 0.33 / 0.22) because their decoders refine boxes as
`sigmoid(inverse_sigmoid(ref) + delta)`, which fp16 cannot resolve.

### Verifying the cards, not just the weights

[`convert/verify_cards.py`](convert/verify_cards.py) — the parity gate proves a
`.pte` matches its PyTorch model and says nothing about whether the preprocessing
and decoding written on the card are right. Those are what an app copies, and a card
documenting the wrong normalisation ships a model nobody can use without any
correlation number noticing. This runs the documented recipe on real images and
asserts something that must hold if it is right: RTMPose's SimCC decode has to put
the nose above the shoulders and the shoulders above the hips, YOLOX's boxes have to
land in-frame with positive area after NMS, MODNet's matte has to be neither empty
nor the whole picture, MoGe's valid pixels have to carry positive finite depth, a confident click on any of
the three SAM-family repos has to return a mask that is neither empty nor the whole
frame, "a photo of a person" has to outrank "a photo of a car" on a portrait through
CLIP's two towers, and Whisper's greedy loop has to terminate at `<|endoftext|>`
instead of running to the window limit. All nine pass.

The CLIP and Whisper checks are the ones that earn their keep: they exercise the
tokenisation and the decode loop, which no parity number touches at all.

It has already earned its place twice. MobileSAM's metadata claimed four mask
outputs where the graph returns three. And DIS shipped with a doubled sigmoid — the
IS-Net forward already applies one — which squashed its mask into [0.5, 0.731], so
the documented "threshold at 0.5" marked every pixel as foreground. Parity read
1.000000 in both cases, because the exported graph faithfully reproduces whatever
wrapper it is handed; only running the documented recipe finds this class of error.

The DIS bug had a second victim. With the output compressed into a third of its
range, thresholded masks agreed too easily, and the int8 audit reported IoU 0.96.
Once the fp32 output was correct the real number was 0.91 median and 0.46 at worst,
and that build has been withdrawn. A broken range upstream flatters every quality
metric downstream.

### Re-authoring helpers

[`convert/fft_ops.py`](convert/fft_ops.py) — **inverse FFT that ExecuTorch can
lower.** `torch.fft.rfftn` exports and runs; the inverse does not. Building a
complex tensor hits `aten.complex`, which is outside the Core ATen opset, and
`view_as_complex` lowers but then fails at runtime. For a fixed spatial size the
inverse is a fixed linear map, so `IRFFT2` precomputes it as two real matmuls —
exportable, delegatable, and 3.1 MB of matrices at 512x512. Verified against
`torch.fft.irfftn` to ~1e-6, and a full Fast Fourier Convolution block round-trips
at corr 1.000000 on both the portable kernels and XNNPACK. This is what unblocks
the LaMa family — big-lama itself converts through it at corr 1.000000. Keep real and imaginary parts as separate tensors so no complex
value ever enters the graph.

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
- **A PReLU model cannot be quantized on 1.4.0** — the two workarounds collide.
  Keeping PReLU on the delegate segfaults (below); excluding it from partitioning
  is fine in fp32 and fails at execute once quantized
  (`Propagating input shapes failed`), because the graph then has portable PReLU
  between delegated quantized convolutions. Three conv+PReLU layers reproduce it.
  Reported on [#21480](https://github.com/pytorch/executorch/pull/21480); landing
  that fix removes the need for the exclusion and closes both paths. This is why
  Real-ESRGAN and TwinLiteNet ship fp32-only.
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
