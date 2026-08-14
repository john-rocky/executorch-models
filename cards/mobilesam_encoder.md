# mobilesam_encoder — ExecuTorch XNNPACK

- **Source**: ChaoningZhang/MobileSAM + dhkim2810/MobileSAM weights
- **License**: Apache-2.0 (code) / MIT (weights)
- **Input**: [[1, 3, 1024, 1024]] — RGB, SAM norm (mean 123.675/116.28/103.53, std 58.395/57.12/57.375), longest side 1024 then pad to 1024x1024
- **Output**: image_embed [1,256,64,64]

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `mobilesam_encoder_xnnpack_fp32.pte` | 28.3 | 1.000000 | 126.1 |
| int8 (dynamic) | `mobilesam_encoder_xnnpack_int8.pte` | 14.0 | 0.999880 | 132.0 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 131.2 ms).

### Checked in the task's own units

Correlation is a first filter. These are the numbers that decide:

- **int8 (dynamic)** — measured in the units that matter for this model — cosine similarity of the image embeddings: median 1.0001 over 10 real images, worst 1.0000.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on random input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 256, 64, 64] | 3.219e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 89.0% (597/671 ops); ops left on the portable kernels: `aten.expand_copy.default` x40, `aten.native_layer_norm.default` x20, `aten.split_with_sizes_copy.default` x10, `aten.mean.dim` x4

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
