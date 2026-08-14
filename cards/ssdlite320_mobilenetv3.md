# ssdlite320_mobilenetv3 — ExecuTorch

- **Source**: torchvision ssdlite320_mobilenet_v3_large COCO_V1
- **License**: BSD-3-Clause
- **Input**: [[1, 3, 320, 320]] — RGB 0-1, 320x320 (torchvision SSDLite norm baked in model)
- **Output**: 12 raw heads: (cls [1,A*91,H,W], box [1,A*4,H,W]) x 6 levels, H=W in {20,10,5,3,2,1}

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `ssdlite320_mobilenetv3_xnnpack_fp32.pte` | 13.8 | 1.000000 | 5.1 |
| int8 | `ssdlite320_mobilenetv3_xnnpack_int8.pte` | 3.9 | 0.968820 | 7.1 |
| Core ML (fp16, iOS) | `ssdlite320_mobilenetv3_coreml_all.pte` | 7.5 | 0.999658 | 0.9 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. Measured on an iPhone 17 Pro across seven models, it
runs **3.5x to 13.9x faster (median 12x)** at roughly half the file size — for
example Depth-Anything-V2-Small at 500.8 ms against 42.7 ms, and MODNet at 81.7 ms
against 5.9 ms. It computes in fp16 and is iOS-only; the XNNPACK files stay the
portable option and are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 115.3 ms).

### Checked in the task's own units

Correlation is a first filter. These are the numbers that decide:

- **int8** — measured in the units that matter for this model — fraction of firing detections agreeing: 0.999 of the fp32 build's detections are matched (26988 of 27004 across 10 images), worst single image 0.998.

### Builds that did not earn a slot

- **fp16 is not shipped**: it comes out at 100% of the fp32 file (13.8 MB vs 13.8 MB), so it buys nothing. XNNPACK serializes convolution weights as fp32 no matter what dtype the graph carries, so on a conv-heavy model fp16 saves no disk and only adds cast operations. Reach for int8 here, not fp16.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 546, 20, 20] | 3.052e-05 | 1.000000 |
| 1 | [1, 24, 20, 20] | 3.767e-05 | 1.000000 |
| 2 | [1, 546, 10, 10] | 2.956e-05 | 1.000000 |
| 3 | [1, 24, 10, 10] | 8.464e-06 | 1.000000 |
| 4 | [1, 546, 5, 5] | 1.717e-05 | 1.000000 |
| 5 | [1, 24, 5, 5] | 9.477e-06 | 1.000000 |
| 6 | [1, 546, 3, 3] | 1.717e-05 | 1.000000 |
| 7 | [1, 24, 3, 3] | 9.421e-06 | 1.000000 |
| 8 | [1, 546, 2, 2] | 2.050e-05 | 1.000000 |
| 9 | [1, 24, 2, 2] | 3.457e-06 | 1.000000 |
| 10 | [1, 546, 1, 1] | 1.001e-05 | 1.000000 |
| 11 | [1, 24, 1, 1] | 9.418e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 94.8% (289/305 ops); ops left on the portable kernels: `dim_order_ops._to_dim_order_copy.default` x16

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
