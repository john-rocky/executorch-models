# depth_anything_v2_small — ExecuTorch XNNPACK

- **Source**: depth-anything/Depth-Anything-V2-Small-hf
- **License**: Apache-2.0
- **Input**: [[1, 3, 518, 518]] — RGB, ImageNet norm, 518x518
- **Output**: relative inverse depth [1,518,518]

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `depth_anything_v2_small_xnnpack_fp32.pte` | 99.0 | 1.000000 | 167.3 |
| fp16 | `depth_anything_v2_small_xnnpack_fp16.pte` | 55.5 | 0.999992 | 289.7 |
| int8 (dynamic) | `depth_anything_v2_small_xnnpack_int8.pte` | 35.5 | 0.999979 | 166.2 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 85.0 ms).

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on random input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 518, 518] | 4.768e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 73.4% (482/657 ops); ops left on the portable kernels: `aten.expand_copy.default` x49, `aten.native_layer_norm.default` x28, `aten.mul.Scalar` x24, `aten.logical_not.default` x24, `aten.eq.Scalar` x12, `aten.full_like.default` x12, `aten.any.dim` x12, `aten.where.self` x12, `dim_order_ops._to_dim_order_copy.default` x1, `aten.squeeze_copy.dims` x1

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
