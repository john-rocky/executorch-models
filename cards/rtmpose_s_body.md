# rtmpose_s_body — ExecuTorch XNNPACK

- **Source**: open-mmlab/mmpose RTMPose-s body7 (rtmpose-s_simcc-body7_pt-body7_420e-256x192)
- **License**: Apache-2.0
- **Input**: [[1, 3, 256, 192]] — RGB, ImageNet norm, 256x192 person crop (detect first, then crop and resize to this aspect)
- **Output**: SimCC pair: x [1,17,384] and y [1,17,512] — a 1-D distribution per keypoint per axis. Decode: keypoint k sits at (argmax(x[k]) / 2, argmax(y[k]) / 2) in crop pixels; the max value doubles as the confidence.

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `rtmpose_s_body_xnnpack_fp32.pte` | 21.9 | 1.000000 | 5.7 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 34.3 ms).

### Precisions that did not earn a slot

- **fp16 is not shipped**: keypoints move a median of 16 px and up to 41 px on a 192x256 crop (169 confident keypoints across 12 person crops). It also only saves 5% of the file, so there is nothing to weigh against that.
- **int8 is not shipped**: measured in the units that matter for this model — largest keypoint displacement against fp32, in crop pixels: median 131.00 over 10 images, worst 163.50.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 17, 384] | 6.855e-06 | 1.000000 |
| 1 | [1, 17, 512] | 8.076e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 92.0% (229/249 ops); ops left on the portable kernels: `dim_order_ops._to_dim_order_copy.default` x8, `aten.unsqueeze_copy.default` x3, `aten.split_with_sizes_copy.default` x3, `aten.sum.dim_IntList` x2, `aten.pow.Tensor_Scalar` x2, `aten.squeeze_copy.dims` x2

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
