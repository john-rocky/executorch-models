# lama_512 — ExecuTorch XNNPACK

- **Source**: advimman/lama + smartywu/big-lama weights
- **License**: Apache-2.0
- **Input**: [[1, 3, 512, 512], [1, 1, 512, 512]] — image RGB 0-1 [1,3,512,512] + mask [1,1,512,512] where 1 marks the region to fill
- **Output**: inpainted image [1,3,512,512] RGB 0-1, already composited with the untouched region

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `lama_512_xnnpack_fp32.pte` | 204.8 | 1.000000 | 833.8 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 638.2 ms).

### Precisions that did not earn a slot

- **int8 is not shipped**: correlation 0.958 clears the 0.95 bar, but for an image-to-image model that bar is the wrong one: 22 dB is visible degradation, where 30 dB reads as near-identical. Not shipped.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 3, 512, 512] | 7.510e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 49.8% (2020/4056 ops); ops left on the portable kernels: `aten.abs.default` x460, `aten.sub.Tensor` x460, `aten.expand_copy.default` x288, `aten.arange.start_step` x230, `aten.index.Tensor` x230, `aten.select_copy.int` x144, `dim_order_ops._to_dim_order_copy.default` x116, `aten.view_as_real_copy.default` x72, `aten._fft_r2c.default` x36

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))

**Notes**: The inverse FFT inside every FourierUnit is replaced with the real matmul form from convert/fft_ops.py; ExecuTorch cannot lower torch.fft.irfftn. Spatial size is fixed because those matrices are built per size.

**Notes (int8)**: The inverse FFT inside every FourierUnit is replaced with the real matmul form from convert/fft_ops.py; ExecuTorch cannot lower torch.fft.irfftn. Spatial size is fixed because those matrices are built per size.
