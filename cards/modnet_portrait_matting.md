# modnet_portrait_matting — ExecuTorch XNNPACK

- **Source**: ZHKKKe/MODNet + DavG25/modnet-pretrained-models ckpt
- **License**: Apache-2.0
- **Input**: [[1, 3, 512, 512]] — RGB [-1,1], 512x512
- **Output**: alpha matte [1,1,512,512] 0-1

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `modnet_portrait_matting_xnnpack_fp32.pte` | 26.1 | 1.000000 | 64.9 |
| fp16 | `modnet_portrait_matting_xnnpack_fp16.pte` | 24.4 | 1.000000 | 107.8 |
| int8 | `modnet_portrait_matting_xnnpack_int8.pte` | 6.8 | 0.999949 | 46.5 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 120.1 ms).

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 1, 512, 512] | 1.476e-04 | 1.000000 |

XNNPACK delegate coverage (fp32): 94.7% (302/319 ops); ops left on the portable kernels: `aten._native_batch_norm_legit.no_stats` x16, `aten.expand_copy.default` x1

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
