# edsr_base_x4 — ExecuTorch

- **Source**: eugenesiow/edsr-base (super-image)
- **License**: Apache-2.0
- **Input**: [[1, 3, 128, 128]] — RGB 0-1, 128x128 tile
- **Output**: SR image [1,3,512,512] RGB, nominally 0-1 but not clamped by the model — it overshoots on high-contrast edges. Clamp before display.

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `edsr_base_x4_xnnpack_fp32.pte` | 6.1 | 1.000000 | 42.5 |
| int8 | `edsr_base_x4_xnnpack_int8.pte` | 1.6 | 0.999918 | 28.3 |
| Core ML (fp16, iOS) | `edsr_base_x4_coreml_all.pte` | 3.3 | 0.999999 | 7.6 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. Measured on an iPhone 17 Pro across seven models, it
runs **3.5x to 13.9x faster (median 12x)** at roughly half the file size — for
example Depth-Anything-V2-Small at 500.8 ms against 42.7 ms, and MODNet at 81.7 ms
against 5.9 ms. It computes in fp16 and is iOS-only; the XNNPACK files stay the
portable option and are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 78.5 ms).

### Checked in the task's own units

Correlation is a first filter. These are the numbers that decide:

- **int8** — measured in the units that matter for this model — PSNR vs the fp32 .pte (dB): median 47.3870 over 10 real images, worst 43.1846.

### Builds that did not earn a slot

- **fp16 is not shipped**: it comes out at 100% of the fp32 file (6.1 MB vs 6.1 MB), so it buys nothing. XNNPACK serializes convolution weights as fp32 no matter what dtype the graph carries, so on a conv-heavy model fp16 saves no disk and only adds cast operations. Reach for int8 here, not fp16.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 3, 512, 512] | 1.788e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 85.7% (96/112 ops); ops left on the portable kernels: `dim_order_ops._to_dim_order_copy.default` x16

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
