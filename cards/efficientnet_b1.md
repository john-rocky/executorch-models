# efficientnet_b1 — ExecuTorch

- **Source**: torchvision efficientnet_b1 IMAGENET1K_V2
- **License**: BSD-3-Clause
- **Input**: [[1, 3, 240, 240]] — RGB, ImageNet norm, 240x240
- **Output**: ImageNet logits [1,1000]

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `efficientnet_b1_xnnpack_fp32.pte` | 31.2 | 1.000000 | 9.3 |
| fp16 | `efficientnet_b1_xnnpack_fp16.pte` | 28.8 | 0.999816 | 54.9 |
| Core ML (fp16, iOS) | `efficientnet_b1_coreml_all.pte` | 16.3 | 0.992817 — see below | 0.6 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. On an iPhone 17 Pro, Depth-Anything-V2-Small runs
500.8 ms through XNNPACK and 42.7 ms through Core ML, at half the file size. It
computes in fp16 and is iOS-only; the XNNPACK files stay the portable option and
are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 352.7 ms).

### Checked in the task's own units

Correlation is a first filter. These are the numbers that decide:

- **Core ML (fp16, iOS)** — measured in the units that matter for this model — fraction of images keeping the fp32 top-1 label: 9 of 10 images keep the fp32 top-1 label.

### Builds that did not earn a slot

- **int8 is not shipped**: measured in the units that matter for this model — fraction of images keeping the fp32 top-1 label: 0 of 10 images keep the fp32 top-1 label.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 1000] | 1.490e-05 | 1.000000 |

XNNPACK delegate coverage (fp32): 100.0% (410/410 ops)

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
