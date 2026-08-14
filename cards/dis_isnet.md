# dis_isnet — ExecuTorch XNNPACK

- **Source**: xuebinqin/DIS + NimaBoscarino/IS-Net_DIS-general-use weights
- **License**: Apache-2.0
- **Input**: [[1, 3, 1024, 1024]] — RGB, scaled to [-1,1] (x/255 then (x-0.5)/0.5), 1024x1024
- **Output**: alpha mask [1,1,1024,1024] 0-1 (sigmoid)

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `dis_isnet_xnnpack_fp32.pte` | 176.1 | 1.000000 | 169.0 |
| int8 | `dis_isnet_xnnpack_int8.pte` | 44.3 | 0.987820 | 70.1 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 392.2 ms).

### Checked in the task's own units

Correlation is a first filter. These are the numbers that decide:

- **int8** — mask IoU 0.963 median against fp32 (worst 0.927 of five images). The disagreement is boundary pixels, which is what int8 costs on a cutout model.

### Precisions that did not earn a slot

- **fp16 is not shipped**: worst-output corr 0.986 against fp32 eager, below the 0.995 bar for this precision. The file converts and runs; the numbers do not hold up, so it is left out rather than shipped with a warning.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 1, 1024, 1024] | 8.941e-07 | 1.000000 |

XNNPACK delegate coverage (fp32): 100.0% (468/468 ops)

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
