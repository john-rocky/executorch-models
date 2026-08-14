# ormbg_isnet — ExecuTorch

- **Source**: schirrmacher/ormbg
- **License**: Apache-2.0
- **Input**: [[1, 3, 1024, 1024]] — RGB 0-1, 1024x1024
- **Output**: alpha mask [1,1,1024,1024] 0-1 (sigmoid)

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `ormbg_isnet_xnnpack_fp32.pte` | 176.1 | 1.000000 | 121.7 |
| int8 | `ormbg_isnet_xnnpack_int8.pte` | 44.3 | 0.999988 | 87.2 |
| Core ML (fp16, iOS) | `ormbg_isnet_coreml_all.pte` | 89.0 | 0.999999 | 28.5 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. Measured on an iPhone 17 Pro across seven models, it
runs **3.5x to 13.9x faster (median 12x)** at roughly half the file size — for
example Depth-Anything-V2-Small at 500.8 ms against 42.7 ms, and MODNet at 81.7 ms
against 5.9 ms. It computes in fp16 and is iOS-only; the XNNPACK files stay the
portable option and are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 375.5 ms).

### Checked in the task's own units

Correlation is a first filter. These are the numbers that decide:

- **int8** — measured in the units that matter for this model — mask IoU at 0.5: median 0.9987 over 10 real images, worst 0.9868.

### Builds that did not earn a slot

- **fp16 is not shipped**: it comes out at 100% of the fp32 file (176.1 MB vs 176.1 MB), so it buys nothing. XNNPACK serializes convolution weights as fp32 no matter what dtype the graph carries, so on a conv-heavy model fp16 saves no disk and only adds cast operations. Reach for int8 here, not fp16.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 1, 1024, 1024] | 2.205e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 100.0% (467/467 ops)

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
