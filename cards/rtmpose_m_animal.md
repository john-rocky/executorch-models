# rtmpose_m_animal — ExecuTorch

- **Source**: open-mmlab/mmpose RTMPose (rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1)
- **License**: Apache-2.0
- **Input**: [[1, 3, 256, 256]] — RGB, ImageNet norm, 256x256 crop around an animal (detect first, then crop and resize to this aspect)
- **Output**: SimCC pair: x [1,17,512] and y [1,17,512] — a 1-D distribution per keypoint per axis. Decode: keypoint k sits at (argmax(x[k]) / 2, argmax(y[k]) / 2) in crop pixels; the max value doubles as the confidence.

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `rtmpose_m_animal_xnnpack_fp32.pte` | 54.5 | 1.000000 | 9.4 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 91.4 ms).

### Builds that did not earn a slot

- **Core ML (fp16, iOS) is not shipped**: measured in the units that matter for this model — fraction of keypoints landing within 4 px of fp32: median 0.9412 over 10 real images, worst 0.7647.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 17, 512] | 2.783e-06 | 1.000000 |
| 1 | [1, 17, 512] | 7.629e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 93.9% (306/326 ops); ops left on the portable kernels: `dim_order_ops._to_dim_order_copy.default` x8, `aten.unsqueeze_copy.default` x3, `aten.split_with_sizes_copy.default` x3, `aten.sum.dim_IntList` x2, `aten.pow.Tensor_Scalar` x2, `aten.squeeze_copy.dims` x2

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))

**Notes**: Top-down again: crop one animal first. AP-10K covers 54 mammal species; a general object detector's animal classes make a workable front end.
