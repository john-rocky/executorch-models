# edsr_base_x4 — ExecuTorch XNNPACK

- **Source**: eugenesiow/edsr-base (super-image)
- **License**: Apache-2.0
- **Input**: [[1, 3, 128, 128]] — RGB 0-1, 128x128 tile
- **Output**: SR image [1,3,512,512] RGB, nominally 0-1 but not clamped by the model — it overshoots on high-contrast edges (measured: 0.7% of pixels outside 0-1, range -0.02 to 1.06 over ten tiles). Clamp before display.

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `edsr_base_x4_xnnpack_fp32.pte` | 6.1 | 1.000000 | 38.9 |
| int8 | `edsr_base_x4_xnnpack_int8.pte` | 1.6 | 0.999918 | 28.3 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 77.9 ms).

### Checked in the task's own units

Correlation is a first filter. These are the numbers that decide:

- **int8** — measured in the units that matter for this model — PSNR vs the fp32 .pte (dB): median 47.3870 over 10 real images, worst 43.1846.

### Precisions that did not earn a slot

- **fp16 is not shipped**: it comes out at 100% of the fp32 file (6.1 MB vs 6.1 MB), so it buys nothing. XNNPACK serializes convolution weights as fp32 no matter what dtype the graph carries, so on a conv-heavy model fp16 saves no disk and only adds cast operations. Reach for int8 here, not fp16.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 3, 512, 512] | 1.788e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 85.7% (96/112 ops); ops left on the portable kernels: `dim_order_ops._to_dim_order_copy.default` x16

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
