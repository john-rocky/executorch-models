# sixdrepnet_headpose — ExecuTorch

- **Source**: thohemp/6DRepNet + osanseviero/6DRepNet_300W_LP_AFLW2000 weights
- **License**: MIT
- **Input**: [[1, 3, 224, 224]] — RGB, ImageNet norm, 224x224 face crop
- **Output**: 6D rotation representation [1,6]. To a rotation matrix: b1 = normalize(v[0:3]); b2 = normalize(v[3:6] - (b1·v[3:6])b1); b3 = cross(b1, b2); R = [b1 b2 b3]. Euler angles follow from R.

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `sixdrepnet_headpose_xnnpack_fp32.pte` | 157.3 | 1.000000 | 7.6 |
| Core ML (fp16, iOS) | `sixdrepnet_headpose_coreml_all.pte` | 78.8 | 0.999991 | 1.5 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. Measured on an iPhone 17 Pro across seven models, it
runs **3.5x to 13.9x faster (median 12x)** at roughly half the file size — for
example Depth-Anything-V2-Small at 500.8 ms against 42.7 ms, and MODNet at 81.7 ms
against 5.9 ms. It computes in fp16 and is iOS-only; the XNNPACK files stay the
portable option and are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 16.3 ms).

### Builds that did not earn a slot

- **fp16 is not shipped**: it comes out at 100% of the fp32 file (157.3 MB vs 157.3 MB), so it buys nothing. XNNPACK serializes convolution weights as fp32 no matter what dtype the graph carries, so on a conv-heavy model fp16 saves no disk and only adds cast operations. Reach for int8 here, not fp16.
- **int8 is not shipped**: worst-output corr 0.815 against fp32 eager, below the 0.95 bar for this precision. The file converts and runs; the numbers do not hold up, so it is left out rather than shipped with a warning.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 6] | 8.643e-07 | 1.000000 |

XNNPACK delegate coverage (fp32): 100.0% (59/59 ops)

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))

**Notes (int8)**: int8 is measured but not shipped. Correlation is a weak read on a six-element output, so the check that matters is the angle between the fp32 and int8 rotations: median 46.5 deg over ten faces, worst 104 deg. That is the known failure mode of re-parameterized RepVGG under post-training quantization — its fused branches leave weight ranges too wide for an int8 grid — and it needs quantization-aware training rather than anything on the export side.
