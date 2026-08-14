# sam21_tiny_decoder — ExecuTorch

- **Source**: facebook/sam2.1-hiera-tiny
- **License**: Apache-2.0
- **Input**: [[1, 256, 64, 64], [1, 32, 256, 256], [1, 64, 128, 128], [1, 1, 1, 2], [1, 1, 1]] — points: pixel coords in 1024x1024 space [1,1,N,2] fp32; labels [1,1,N] int64 (1=fg, 0=bg); prompt encoder embedded
- **Output**: mask logits [1,1,3,256,256] (upsample 4x to 1024, >0 = fg), iou [1,1,3]

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `sam21_tiny_decoder_xnnpack_fp32.pte` | 24.7 | 1.000000 | 20.0 |
| fp16 | `sam21_tiny_decoder_xnnpack_fp16.pte` | 12.6 | 0.999999 | 52.1 |
| Core ML (fp16, iOS) | `sam21_tiny_decoder_coreml_all.pte` | 12.7 | 0.999999 | 3.5 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. On an iPhone 17 Pro, Depth-Anything-V2-Small runs
500.8 ms through XNNPACK and 42.7 ms through Core ML, at half the file size. It
computes in fp16 and is iOS-only; the XNNPACK files stay the portable option and
are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 9.1 ms).

### Builds that did not earn a slot

- **int8 (dynamic) is not shipped**: at 12.9 MB it is no smaller than the fp16 build, which is also the more faithful of the two. Nothing would pick it.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on random input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 1, 3, 256, 256] | 2.241e-05 | 1.000000 |
| 1 | [1, 1, 3] | 7.749e-07 | 1.000000 |

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
