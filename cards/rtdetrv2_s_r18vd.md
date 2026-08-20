# rtdetrv2_s_r18vd — ExecuTorch

- **Source**: PekingU/rtdetr_v2_r18vd
- **License**: Apache-2.0
- **Input**: [[1, 3, 640, 640]] — RGB/255 only (no mean/std norm), 640x640
- **Output**: logits [1,300,80] (sigmoid -> per-class score), boxes [1,300,4] cxcywh normalized 0..1; postprocess = sigmoid + top-k, NO NMS

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `rtdetrv2_s_r18vd_xnnpack_fp32.pte` | 80.9 | 1.000000 | 56.5 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 64.9 ms).

### Builds that did not earn a slot

- **fp16 is not shipped**: worst-output corr 0.329 against fp32 eager, below the 0.995 bar for this precision. The file converts and runs; the numbers do not hold up, so it is left out rather than shipped with a warning.
- **Core ML (fp16, iOS) is not shipped**: worst-output corr 0.217 against fp32 eager, below the 0.995 bar for this precision. The file converts and runs; the numbers do not hold up, so it is left out rather than shipped with a warning.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 300, 80] | 2.992e-04 | 1.000000 |
| 1 | [1, 300, 4] | 3.764e-05 | 1.000000 |

XNNPACK delegate coverage (fp32): 75.9% (839/1106 ops); ops left on the portable kernels: `aten.expand_copy.default` x36, `dim_order_ops._to_dim_order_copy.default` x26, `aten.arange.start_step` x21, `aten.select_copy.int` x20, `aten.view_copy.default` x18, `aten.where.self` x17, `aten.eq.Scalar` x16, `aten.unsqueeze_copy.default` x15, `aten.native_layer_norm.default` x12, `aten.logical_not.default` x10, `aten.grid_sampler_2d.default` x9, `aten.mul.Scalar` x8, `aten.full_like.default` x7, `aten.copy.default` x6, `aten.split_with_sizes_copy.default` x6, `aten.any.dim` x5, `aten.mul.Tensor` x4, `aten.alias_copy.default` x4, `aten.avg_pool2d.default` x3, `aten.sum.dim_IntList` x3, `dim_order_ops._clone_dim_order.default` x2, `aten.sin.default` x2, `aten.cos.default` x2, `aten.upsample_nearest2d.vec` x2, `aten.repeat.default` x2, `aten.gather.default` x2, `aten.full.default` x1, `aten.div.Tensor` x1, `aten.pow.Scalar` x1, `aten.reciprocal.default` x1, `aten.cat.default` x1, `aten.gt.Scalar` x1, `aten.lt.Scalar` x1, `aten.max.dim` x1, `aten.topk.default` x1

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
