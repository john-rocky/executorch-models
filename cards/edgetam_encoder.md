# edgetam_encoder — ExecuTorch

- **Source**: yonigozlan/EdgeTAM-hf
- **License**: Apache-2.0
- **Input**: [[1, 3, 1024, 1024]] — RGB/255, imagenet norm (mean .485/.456/.406, std .229/.224/.225), 1024x1024
- **Output**: image_embed [1,256,64,64], feat_s0 [1,32,256,256], feat_s1 [1,64,128,128]

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `edgetam_encoder_xnnpack_fp32.pte` | 19.7 | 1.000000 | 32.3 |
| Core ML (fp16, iOS) | `edgetam_encoder_coreml_all.pte` | 10.6 | 0.999822 | 9.6 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. On an iPhone 17 Pro, Depth-Anything-V2-Small runs
500.8 ms through XNNPACK and 42.7 ms through Core ML, at half the file size. It
computes in fp16 and is iOS-only; the XNNPACK files stay the portable option and
are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 103.5 ms).

### Builds that did not earn a slot

- **fp16 is not shipped**: it comes out at 101% of the fp32 file (19.8 MB vs 19.7 MB), so it buys nothing. XNNPACK serializes convolution weights as fp32 no matter what dtype the graph carries, so on a conv-heavy model fp16 saves no disk and only adds cast operations. Reach for int8 here, not fp16.
- **int8 (dynamic) is not shipped**: it comes out at 100% of the fp32 file (19.7 MB vs 19.7 MB), so it buys nothing. XNNPACK serializes convolution weights as fp32 no matter what dtype the graph carries, so on a conv-heavy model fp16 saves no disk and only adds cast operations. Reach for int8 here, not fp16.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on random input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 256, 64, 64] | 2.509e-05 | 1.000000 |
| 1 | [1, 32, 256, 256] | 8.643e-06 | 1.000000 |
| 2 | [1, 64, 128, 128] | 1.499e-05 | 1.000000 |

XNNPACK delegate coverage (fp32): 99.8% (399/400 ops); ops left on the portable kernels: `aten.upsample_nearest2d.vec` x1

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
