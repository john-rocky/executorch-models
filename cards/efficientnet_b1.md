# efficientnet_b1 — ExecuTorch XNNPACK

- **Source**: torchvision efficientnet_b1 IMAGENET1K_V2
- **License**: BSD-3-Clause
- **Input**: [[1, 3, 240, 240]] — RGB, ImageNet norm, 240x240
- **Output**: ImageNet logits [1,1000]

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `efficientnet_b1_xnnpack_fp32.pte` | 31.2 | 1.000000 | 9.3 |
| fp16 | `efficientnet_b1_xnnpack_fp16.pte` | 28.8 | 0.999816 | 54.9 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 352.7 ms).

### Precisions that did not earn a slot

- **int8 is not shipped**: worst-output corr 0.077 against fp32 eager, below the 0.95 bar for this precision. The file converts and runs; the numbers do not hold up, so it is left out rather than shipped with a warning.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 1000] | 1.490e-05 | 1.000000 |

XNNPACK delegate coverage (fp32): 100.0% (410/410 ops)

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
