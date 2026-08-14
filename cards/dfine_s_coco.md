# dfine_s_coco — ExecuTorch

- **Source**: ustc-community/dfine-small-coco
- **License**: Apache-2.0
- **Input**: [[1, 3, 640, 640]] — RGB/255 only (no mean/std norm), 640x640
- **Output**: logits [1,300,80] (sigmoid -> per-class score), boxes [1,300,4] cxcywh normalized 0..1; postprocess = sigmoid + top-k, NO NMS

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `dfine_s_coco_xnnpack_fp32.pte` | 41.5 | 1.000000 | 54.0 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 138.0 ms).

### Builds that did not earn a slot

- **fp16 is not shipped**: worst-output corr 0.223 against fp32 eager, below the 0.995 bar for this precision. The file converts and runs; the numbers do not hold up, so it is left out rather than shipped with a warning.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 300, 80] | 1.420e-03 | 1.000000 |
| 1 | [1, 300, 4] | 4.858e-05 | 1.000000 |

XNNPACK delegate coverage (fp32): 77.5% (1391/1794 ops); ops left on the portable kernels: `dim_order_ops._to_dim_order_copy.default` x64, `aten.select_copy.int` x59, `aten.expand_copy.default` x36, `aten.pow.Tensor_Scalar` x29, `aten.arange.start_step` x21, `aten.view_copy.default` x18, `aten.where.self` x17, `aten.eq.Scalar` x16, `aten.unsqueeze_copy.default` x15, `aten.split_with_sizes_copy.default` x13, `aten.native_layer_norm.default` x12, `aten.squeeze_copy.dims` x12, `aten.logical_not.default` x10, `aten.alias_copy.default` x10, `aten.grid_sampler_2d.default` x9, `aten.mul.Scalar` x8, `aten.full_like.default` x8, `aten.copy.default` x6, `aten.sum.dim_IntList` x6, `aten.any.dim` x5, `aten.mul.Tensor` x4, `aten.cat.default` x3, `dim_order_ops._clone_dim_order.default` x2, `aten.sin.default` x2, `aten.cos.default` x2, `aten.upsample_nearest2d.vec` x2, `aten.topk.default` x2, `aten.repeat.default` x2, `aten.gather.default` x2, `aten.full.default` x1, `aten.div.Tensor` x1, `aten.pow.Scalar` x1, `aten.reciprocal.default` x1, `aten.gt.Scalar` x1, `aten.lt.Scalar` x1, `aten.max.dim` x1, `aten.mean.dim` x1

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
