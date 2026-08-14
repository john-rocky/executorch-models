# dis_isnet — ExecuTorch

- **Source**: xuebinqin/DIS + NimaBoscarino/IS-Net_DIS-general-use weights
- **License**: Apache-2.0
- **Input**: [[1, 3, 1024, 1024]] — RGB, scaled to [-1,1] (x/255 then (x-0.5)/0.5), 1024x1024
- **Output**: alpha mask [1,1,1024,1024] 0-1 (sigmoid)

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `dis_isnet_xnnpack_fp32.pte` | 176.1 | 1.000000 | 123.4 |
| Core ML (fp16, iOS) | `dis_isnet_coreml_all.pte` | 89.0 | 0.999984 | 29.2 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. On an iPhone 17 Pro, Depth-Anything-V2-Small runs
500.8 ms through XNNPACK and 42.7 ms through Core ML, at half the file size. It
computes in fp16 and is iOS-only; the XNNPACK files stay the portable option and
are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 364.6 ms).

### Builds that did not earn a slot

- **fp16 is not shipped**: worst-output corr 0.986 against fp32 eager, below the 0.995 bar for this precision. The file converts and runs; the numbers do not hold up, so it is left out rather than shipped with a warning.
- **int8 is not shipped**: measured in the units that matter for this model — mask IoU at 0.5: median 0.9106 over 10 real images, worst 0.4647.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 1, 1024, 1024] | 3.606e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 100.0% (467/467 ops)

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
