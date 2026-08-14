# mobilesam_decoder — ExecuTorch

- **Source**: ChaoningZhang/MobileSAM + dhkim2810/MobileSAM weights
- **License**: Apache-2.0 (code) / MIT (weights)
- **Input**: [[1, 256, 64, 64], [1, 1, 2], [1, 1]] — points: pixel coords in the padded 1024x1024 space [1,N,2] fp32; labels [1,N] fp32 (1=fg, 0=bg, -1=pad); prompt encoder embedded
- **Output**: mask logits [1,3,256,256] (upsample 4x to 1024, >0 = fg), iou [1,3]

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `mobilesam_decoder_xnnpack_fp32.pte` | 20.5 | 1.000000 | 20.0 |
| fp16 | `mobilesam_decoder_xnnpack_fp16.pte` | 10.5 | 0.999825 | 31.0 |
| Core ML (fp16, iOS) | `mobilesam_decoder_coreml_all.pte` | 16.9 | 0.999999 | 3.0 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. Measured on an iPhone 17 Pro across seven models, it
runs **3.5x to 13.9x faster (median 12x)** at roughly half the file size — for
example Depth-Anything-V2-Small at 500.8 ms against 42.7 ms, and MODNet at 81.7 ms
against 5.9 ms. It computes in fp16 and is iOS-only; the XNNPACK files stay the
portable option and are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 9.9 ms).

### Builds that did not earn a slot

- **int8 (dynamic) is not shipped**: it comes out at 106% of the fp32 file (21.8 MB vs 20.5 MB), so it buys nothing. XNNPACK serializes convolution weights as fp32 no matter what dtype the graph carries, so on a conv-heavy model fp16 saves no disk and only adds cast operations. Reach for int8 here, not fp16.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on random input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 3, 256, 256] | 1.335e-05 | 1.000000 |
| 1 | [1, 3] | 0.000e+00 | 1.000000 |

XNNPACK delegate coverage (fp32): 80.6% (279/346 ops); ops left on the portable kernels: `aten.expand_copy.default` x32, `aten.native_layer_norm.default` x9, `dim_order_ops._to_dim_order_copy.default` x8, `aten.select_copy.int` x7, `aten.unsqueeze_copy.default` x4, `aten.eq.Scalar` x3, `aten.full.default` x2, `aten.mean.dim` x2

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
