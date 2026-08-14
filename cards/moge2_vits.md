# moge2_vits — ExecuTorch XNNPACK

- **Source**: Ruicheng/moge-2-vits-normal
- **License**: MIT
- **Input**: [[1, 3, 518, 518]] — RGB, ImageNet norm, 518x518
- **Output**: points [1,H,W,3] metric point map, normal [1,H,W,3], mask [1,H,W] validity, metric_scale [1]

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `moge2_vits_xnnpack_fp32.pte` | 140.8 | 0.999998 | 750.9 |
| fp16 | `moge2_vits_xnnpack_fp16.pte` | 96.6 | 0.434851 — see below | 1714.8 |
| int8 (dynamic) | `moge2_vits_xnnpack_int8.pte` | 76.4 | 0.998758 | 746.5 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 345.4 ms).

### Checked in the task's own units

Correlation is a first filter. These are the numbers that decide:

- **fp16** — correlation reads 0.43 on this build, and that number is an artifact: one of the four outputs is a near-binary validity mask whose raw logits correlate badly while the thresholded mask is identical. Measured properly against fp32 — mask IoU 1.0000, point map cosine 1.000000, normals cosine 1.000000, metric scale within 0.6% — the geometry is unchanged.
- **int8 (dynamic)** — point map and normals hold at cosine 0.999 against the fp32 build.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 518, 518, 3] | 1.431e-06 | 1.000000 |
| 1 | [1, 518, 518, 3] | 1.878e-06 | 1.000000 |
| 2 | [1, 518, 518] | 1.192e-07 | 0.999998 |
| 3 | [1] | 5.960e-07 | nan |

XNNPACK delegate coverage (fp32): 53.7% (623/1160 ops); ops left on the portable kernels: `aten.arange.start_step` x90, `aten.clamp.default` x80, `aten.index.Tensor` x80, `aten.expand_copy.default` x65, `aten.squeeze_copy.dims` x38, `aten.native_layer_norm.default` x26, `aten.mul.Scalar` x24, `aten.logical_not.default` x24, `aten.where.self` x22, `dim_order_ops._to_dim_order_copy.default` x20, `aten.eq.Scalar` x12, `aten.full_like.default` x12, `aten.any.dim` x12, `aten.lt.Scalar` x10, `aten.sub.Tensor` x10, `aten.unsqueeze_copy.default` x5, `aten.sum.dim_IntList` x2, `aten.pow.Tensor_Scalar` x2, `aten._upsample_bilinear2d_aa.default` x1, `aten.select_copy.int` x1, `aten.split_with_sizes_copy.default` x1

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
