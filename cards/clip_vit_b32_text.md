# clip_vit_b32_text — ExecuTorch

- **Source**: openai/clip-vit-base-patch32
- **License**: MIT
- **Input**: [[1, 77], [1, 77]] — CLIP BPE tokens, fixed len 77 with attention mask
- **Output**: text embedding [1,512] (unnormalized; L2-normalize before cosine)

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `clip_vit_b32_text_xnnpack_fp32.pte` | 253.9 | 1.000000 | 18.7 |
| fp16 | `clip_vit_b32_text_xnnpack_fp16.pte` | 127.1 | 1.000000 | 26.4 |
| Core ML (fp16, iOS) | `clip_vit_b32_text_coreml_all.pte` | 127.4 | 1.000000 | 2.5 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. On an iPhone 17 Pro, Depth-Anything-V2-Small runs
500.8 ms through XNNPACK and 42.7 ms through Core ML, at half the file size. It
computes in fp16 and is iOS-only; the XNNPACK files stay the portable option and
are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 15.6 ms).

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on random input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 512] | 1.907e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 63.0% (398/632 ops); ops left on the portable kernels: `aten.expand_copy.default` x49, `aten.native_layer_norm.default` x25, `aten.scalar_tensor.default` x24, `aten.mul.Scalar` x24, `aten.where.self` x24, `aten.logical_not.default` x24, `aten.eq.Scalar` x12, `aten.full_like.default` x12, `aten.any.dim` x12, `aten.unsqueeze_copy.default` x9, `aten.arange.start_step` x4, `dim_order_ops._to_dim_order_copy.default` x2, `aten.add.Tensor` x2, `aten.embedding.default` x2, `aten.index.Tensor` x2, `aten.bitwise_and.Tensor` x2, `aten.full.default` x1, `aten.alias_copy.default` x1, `aten.view_copy.default` x1, `aten.argmax.default` x1, `aten.le.Tensor` x1

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
