# rtmpose_m_face — ExecuTorch XNNPACK

- **Source**: open-mmlab/mmpose RTMPose (rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400)
- **License**: Apache-2.0
- **Input**: [[1, 3, 256, 256]] — RGB, ImageNet norm, 256x256 crop around a face (detect first, then crop and resize to this aspect)
- **Output**: SimCC pair: x [1,106,512] and y [1,106,512] — a 1-D distribution per keypoint per axis. Decode: keypoint k sits at (argmax(x[k]) / 2, argmax(y[k]) / 2) in crop pixels; the max value doubles as the confidence.

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `rtmpose_m_face_xnnpack_fp32.pte` | 67.9 | 1.000000 | 10.4 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 94.4 ms).

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 106, 512] | 1.147e-05 | 1.000000 |
| 1 | [1, 106, 512] | 9.179e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 93.9% (306/326 ops); ops left on the portable kernels: `dim_order_ops._to_dim_order_copy.default` x8, `aten.unsqueeze_copy.default` x3, `aten.split_with_sizes_copy.default` x3, `aten.sum.dim_IntList` x2, `aten.pow.Tensor_Scalar` x2, `aten.squeeze_copy.dims` x2

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))

**Notes**: Top-down: crop one face first. In practice a portrait framed on the head works directly, which is how the card check exercises it.
