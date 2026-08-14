# real_esrgan_x4v3 — ExecuTorch XNNPACK

- **Source**: xinntao/Real-ESRGAN release v0.2.5.0 (realesr-general-x4v3)
- **License**: BSD-3-Clause
- **Input**: [[1, 3, 128, 128]] — RGB 0-1, 128x128 tile
- **Output**: SR image [1,3,512,512] RGB 0-1

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `real_esrgan_x4v3_xnnpack_fp32.pte` | 4.9 | 1.000000 | 33.9 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 42.3 ms).

### Precisions that did not earn a slot

- **fp16 is not shipped**: it comes out at 100% of the fp32 file (4.9 MB vs 4.9 MB), so it buys nothing. XNNPACK serializes convolution weights as fp32 no matter what dtype the graph carries, so on a conv-heavy model fp16 saves no disk and only adds cast operations. Reach for int8 here, not fp16.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 3, 512, 512] | 1.235e-05 | 1.000000 |

XNNPACK delegate coverage (fp32): 61.3% (106/173 ops); ops left on the portable kernels: `aten.gt.Scalar` x33, `aten.where.self` x33, `aten.upsample_nearest2d.vec` x1

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
