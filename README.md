# ExecuTorch-Models

Converted **.pte model zoo** for iOS / Android / edge with [ExecuTorch](https://pytorch.org/executorch/) XNNPACK.

Every model ships on Hugging Face with a verification card: parity vs PyTorch fp32
(max_abs_diff + correlation), measured latency, exact pre/post-processing spec, and license.
Conversion scripts for all models are in [`convert/`](convert/) — one common harness,
one small script per model.

**If you like this repository, please give it a star.**

## Models

### Vision

| Model | Task | .pte | Parity (corr) | License |
|-------|------|------|--------------|---------|
| [SAM2.1-hiera-tiny](https://huggingface.co/mlboydaisuke/SAM2.1-hiera-tiny-ExecuTorch) | promptable segmentation | 109 MB enc + 25 MB dec | 1.000000 (E2E mask IoU 1.0000) | Apache-2.0 |
| [RT-DETRv2-S](https://huggingface.co/mlboydaisuke/RT-DETRv2-S-ExecuTorch) | object detection (no NMS) | 81 MB | 1.000000 | Apache-2.0 |
| [D-FINE-S](https://huggingface.co/mlboydaisuke/D-FINE-S-ExecuTorch) | object detection (no NMS) | 42 MB | 1.000000 | Apache-2.0 |
| [YOLOX-s](https://huggingface.co/mlboydaisuke/YOLOX-s-ExecuTorch) | object detection | 36 MB | 1.000000 | Apache-2.0 |
| [SSDLite320-MobileNetV3](https://huggingface.co/mlboydaisuke/SSDLite320-MobileNetV3-ExecuTorch) | object detection (raw head) | 14 MB | 1.000000 | BSD-3 |
| [Depth-Anything-V2-Small](https://huggingface.co/mlboydaisuke/Depth-Anything-V2-Small-ExecuTorch) | monocular depth | 99 MB | 1.000000 | Apache-2.0 |
| [DINOv2 ViT-S/14](https://huggingface.co/mlboydaisuke/DINOv2-ViT-S14-ExecuTorch) | feature extraction | 88 MB | 1.000000 | Apache-2.0 |
| [CLIP ViT-B/32](https://huggingface.co/mlboydaisuke/CLIP-ViT-B32-ExecuTorch) | zero-shot classification | image + text towers | 1.000000 | MIT |
| [MODNet](https://huggingface.co/mlboydaisuke/MODNet-ExecuTorch) | portrait matting | 26 MB | 1.000000 | Apache-2.0 |
| [ormbg (ISNet)](https://huggingface.co/mlboydaisuke/ormbg-ExecuTorch) | background removal | 176 MB | 1.000000 | Apache-2.0 |
| [PIDNet-S](https://huggingface.co/mlboydaisuke/PIDNet-S-Cityscapes-ExecuTorch) | semantic segmentation | 31 MB | 1.000000 | MIT |
| [TwinLiteNet](https://huggingface.co/mlboydaisuke/TwinLiteNet-ExecuTorch) | drivable area + lanes | 2 MB | 1.000000 | MIT |
| [EDSR ×4](https://huggingface.co/mlboydaisuke/EDSR-x4-ExecuTorch) | super-resolution | 6 MB | 1.000000 | Apache-2.0 |
| [EfficientNet-B1](https://huggingface.co/mlboydaisuke/EfficientNet-B1-ExecuTorch) | classification | 31 MB | 1.000000 | BSD-3 |

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
`results/*.json`, and [`convert/gen_cards.py`](convert/gen_cards.py) renders the
cards. Per-model scripts are thin wrappers that:

- unwrap dict/dataclass outputs to tensors (harness accepts tensors/tuples only)
- tap raw detection heads instead of NMS-bearing forward paths
- fix layout at graph boundaries (`.contiguous()` — see SAM2.1 notes below)

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

## Environment

- executorch 1.4.0 (pip), torch 2.13, Python 3.12
- LLM params JSONs are not bundled in the pip wheel — copy from the
  [executorch repo](https://github.com/pytorch/executorch) `examples/models/<family>/config/`
  at the matching tag.
