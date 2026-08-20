# lama_512 — ExecuTorch

- **Source**: advimman/lama + smartywu/big-lama weights
- **License**: Apache-2.0
- **Input**: [[1, 3, 512, 512], [1, 1, 512, 512]] — image RGB 0-1 [1,3,512,512] + mask [1,1,512,512] where 1 marks the region to fill
- **Output**: inpainted image [1,3,512,512] RGB 0-1, already composited with the untouched region

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `lama_512_xnnpack_fp32.pte` | 205.0 | 1.000000 | 809.7 |
| Core ML (fp16, iOS) | `lama_512_coreml_all.pte` | 105.3 | 0.999553 | 59.5 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. Measured on an iPhone 17 Pro across seven models, it
runs **3.5x to 13.9x faster (median 12x)** at roughly half the file size — for
example Depth-Anything-V2-Small at 500.8 ms against 42.7 ms, and MODNet at 81.7 ms
against 5.9 ms. It computes in fp16 and is iOS-only; the XNNPACK files stay the
portable option and are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 521.4 ms).

### Builds that did not earn a slot

- **int8 is not shipped**: measured in the units that matter for this model — PSNR vs the fp32 .pte (dB): median 22.4666 over 5 real images, worst 21.3621.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 3, 512, 512] | 7.629e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 57.4% (2884/5028 ops); ops left on the portable kernels: `aten.expand_copy.default` x576, `aten.abs.default` x460, `aten.sub.Tensor` x460, `aten.arange.start_step` x230, `aten.index.Tensor` x230, `dim_order_ops._to_dim_order_copy.default` x116, `aten.select_copy.int` x72

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))

**Notes**: The inverse FFT inside every FourierUnit is replaced with the real matmul form from convert/fft_ops.py; ExecuTorch cannot lower torch.fft.irfftn. Spatial size is fixed because those matrices are built per size.

**Notes (int8)**: The inverse FFT inside every FourierUnit is replaced with the real matmul form from convert/fft_ops.py; ExecuTorch cannot lower torch.fft.irfftn. Spatial size is fixed because those matrices are built per size.

**Notes (coreml_all)**: The inverse FFT inside every FourierUnit is replaced with the real matmul form from convert/fft_ops.py; ExecuTorch cannot lower torch.fft.irfftn. Spatial size is fixed because those matrices are built per size.
