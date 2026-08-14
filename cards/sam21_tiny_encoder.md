# sam21_tiny_encoder — ExecuTorch

- **Source**: facebook/sam2.1-hiera-tiny
- **License**: Apache-2.0
- **Input**: [[1, 3, 1024, 1024]] — RGB/255, imagenet norm (mean .485/.456/.406, std .229/.224/.225), 1024x1024
- **Output**: image_embed [1,256,64,64], feat_s0 [1,32,256,256], feat_s1 [1,64,128,128]

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `sam21_tiny_encoder_xnnpack_fp32.pte` | 109.2 | 1.000000 | 2506.3 |
| fp16 | `sam21_tiny_encoder_xnnpack_fp16.pte` | 55.6 | 0.999993 | 3041.1 |
| Core ML (fp16, iOS) | `sam21_tiny_encoder_coreml_all.pte` | 70.0 | 0.999904 | 156.1 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. Measured on an iPhone 17 Pro across seven models, it
runs **3.5x to 13.9x faster (median 12x)** at roughly half the file size — for
example Depth-Anything-V2-Small at 500.8 ms against 42.7 ms, and MODNet at 81.7 ms
against 5.9 ms. It computes in fp16 and is iOS-only; the XNNPACK files stay the
portable option and are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 248.6 ms).

### Builds that did not earn a slot

- **int8 (dynamic) is not shipped**: it comes out at 100% of the fp32 file (109.2 MB vs 109.2 MB), so it buys nothing. XNNPACK serializes convolution weights as fp32 no matter what dtype the graph carries, so on a conv-heavy model fp16 saves no disk and only adds cast operations. Reach for int8 here, not fp16.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on random input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 256, 64, 64] | 7.987e-06 | 1.000000 |
| 1 | [1, 32, 256, 256] | 5.960e-07 | 1.000000 |
| 2 | [1, 64, 128, 128] | 2.027e-06 | 1.000000 |

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
