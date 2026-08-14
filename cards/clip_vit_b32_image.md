# clip_vit_b32_image — ExecuTorch XNNPACK

- **Source**: openai/clip-vit-base-patch32
- **License**: MIT
- **Input**: [[1, 3, 224, 224]] — RGB, CLIP norm (mean .481/.458/.408, std .269/.261/.276), 224x224
- **Output**: image embedding [1,512] (unnormalized; L2-normalize before cosine)

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `clip_vit_b32_image_xnnpack_fp32.pte` | 351.6 | 1.000000 | 19.0 |
| fp16 | `clip_vit_b32_image_xnnpack_fp16.pte` | 180.7 | 0.999996 | 26.1 |
| int8 (dynamic) | `clip_vit_b32_image_xnnpack_int8.pte` | 95.9 | 0.995739 | 18.4 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 18.5 ms).

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 512] | 9.179e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 69.3% (390/563 ops); ops left on the portable kernels: `aten.expand_copy.default` x49, `aten.native_layer_norm.default` x26, `aten.mul.Scalar` x24, `aten.logical_not.default` x24, `aten.eq.Scalar` x12, `aten.full_like.default` x12, `aten.any.dim` x12, `aten.where.self` x12, `aten.embedding.default` x1, `aten.select_copy.int` x1

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
