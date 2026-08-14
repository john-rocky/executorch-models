# depth_anything_v2_small — ExecuTorch

- **Source**: depth-anything/Depth-Anything-V2-Small-hf
- **License**: Apache-2.0
- **Input**: [[1, 3, 518, 518]] — RGB, ImageNet norm, 518x518
- **Output**: relative inverse depth [1,518,518]

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `depth_anything_v2_small_xnnpack_fp32.pte` | 99.0 | 1.000000 | 167.3 |
| fp16 | `depth_anything_v2_small_xnnpack_fp16.pte` | 55.5 | 0.999992 | 289.7 |
| Core ML (fp16, iOS) | `depth_anything_v2_small_coreml_all.pte` | 50.2 | 0.999992 | 48.1 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. Measured on an iPhone 17 Pro across seven models, it
runs **3.5x to 13.9x faster (median 12x)** at roughly half the file size — for
example Depth-Anything-V2-Small at 500.8 ms against 42.7 ms, and MODNet at 81.7 ms
against 5.9 ms. It computes in fp16 and is iOS-only; the XNNPACK files stay the
portable option and are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 85.0 ms).

### Builds that did not earn a slot

- **int8 (dynamic) is not shipped**: measured in the units that matter for this model — fraction of pixels within 1.25x of the fp32 depth: median 0.9941 over 10 real images, worst 0.9748.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on random input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 518, 518] | 4.768e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 73.4% (482/657 ops); ops left on the portable kernels: `aten.expand_copy.default` x49, `aten.native_layer_norm.default` x28, `aten.mul.Scalar` x24, `aten.logical_not.default` x24, `aten.eq.Scalar` x12, `aten.full_like.default` x12, `aten.any.dim` x12, `aten.where.self` x12, `dim_order_ops._to_dim_order_copy.default` x1, `aten.squeeze_copy.dims` x1

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
