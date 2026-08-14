# modnet_portrait_matting — ExecuTorch

- **Source**: ZHKKKe/MODNet + DavG25/modnet-pretrained-models ckpt
- **License**: Apache-2.0
- **Input**: [[1, 3, 512, 512]] — RGB [-1,1], 512x512
- **Output**: alpha matte [1,1,512,512] 0-1

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `modnet_portrait_matting_xnnpack_fp32.pte` | 26.1 | 1.000000 | 64.9 |
| fp16 | `modnet_portrait_matting_xnnpack_fp16.pte` | 24.4 | 1.000000 | 107.8 |
| Core ML (fp16, iOS) | `modnet_portrait_matting_coreml_all.pte` | 13.8 | 0.999997 | 6.8 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. Measured on an iPhone 17 Pro across seven models, it
runs **3.5x to 13.9x faster (median 12x)** at roughly half the file size — for
example Depth-Anything-V2-Small at 500.8 ms against 42.7 ms, and MODNet at 81.7 ms
against 5.9 ms. It computes in fp16 and is iOS-only; the XNNPACK files stay the
portable option and are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 120.1 ms).

### Builds that did not earn a slot

- **int8 is not shipped**: measured in the units that matter for this model — mask IoU at 0.5: median 0.9864 over 10 real images, worst 0.5970.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 1, 512, 512] | 1.476e-04 | 1.000000 |

XNNPACK delegate coverage (fp32): 94.7% (302/319 ops); ops left on the portable kernels: `aten._native_batch_norm_legit.no_stats` x16, `aten.expand_copy.default` x1

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
