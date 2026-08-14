# edgetam_decoder — ExecuTorch

- **Source**: yonigozlan/EdgeTAM-hf
- **License**: Apache-2.0
- **Input**: [[1, 256, 64, 64], [1, 32, 256, 256], [1, 64, 128, 128], [1, 1, 1, 2], [1, 1, 1]] — points: pixel coords in 1024x1024 space [1,1,N,2] fp32; labels [1,1,N] int64 (1=fg, 0=bg); prompt encoder embedded
- **Output**: mask logits [1,1,3,256,256] (upsample 4x to 1024, >0 = fg), iou [1,1,3]

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `edgetam_decoder_xnnpack_fp32.pte` | 24.7 | 1.000000 | 23.7 |
| fp16 | `edgetam_decoder_xnnpack_fp16.pte` | 12.6 | 1.000000 | 51.3 |
| Core ML (fp16, iOS) | `edgetam_decoder_coreml_all.pte` | 12.7 | 0.999998 | 6.5 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. Measured on an iPhone 17 Pro across seven models, it
runs **3.5x to 13.9x faster (median 12x)** at roughly half the file size — for
example Depth-Anything-V2-Small at 500.8 ms against 42.7 ms, and MODNet at 81.7 ms
against 5.9 ms. It computes in fp16 and is iOS-only; the XNNPACK files stay the
portable option and are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 13.9 ms).

### Builds that did not earn a slot

- **int8 (dynamic) is not shipped**: at 12.9 MB it is no smaller than the fp16 build, which is also the more faithful of the two. Nothing would pick it.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on random input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 1, 3, 256, 256] | 4.435e-05 | 1.000000 |
| 1 | [1, 1, 3] | 3.129e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 66.5% (272/409 ops); ops left on the portable kernels: `aten.expand_copy.default` x32, `aten.mul.Scalar` x14, `aten.logical_not.default` x14, `aten.where.self` x11, `aten.eq.Scalar` x10, `aten.native_layer_norm.default` x10, `aten.select_copy.int` x9, `aten.full_like.default` x8, `aten.any.dim` x7, `dim_order_ops._to_dim_order_copy.default` x5, `aten.unsqueeze_copy.default` x5, `aten.arange.start_step` x2, `aten.view_copy.default` x2, `aten.copy.default` x2, `aten.constant_pad_nd.default` x1, `aten.clamp.default` x1, `aten.ge.Scalar` x1, `aten.ne.Scalar` x1, `aten.embedding.default` x1, `aten.repeat.default` x1

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
