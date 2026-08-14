# rtmpose_m_hand — ExecuTorch

- **Source**: open-mmlab/mmpose RTMPose (rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594)
- **License**: Apache-2.0
- **Input**: [[1, 3, 256, 256]] — RGB, ImageNet norm, 256x256 crop around a hand (detect first, then crop and resize to this aspect)
- **Output**: SimCC pair: x [1,21,512] and y [1,21,512] — a 1-D distribution per keypoint per axis. Decode: keypoint k sits at (argmax(x[k]) / 2, argmax(y[k]) / 2) in crop pixels; the max value doubles as the confidence.

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `rtmpose_m_hand_xnnpack_fp32.pte` | 55.1 | 1.000000 | 9.6 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 94.8 ms).

### Builds that did not earn a slot

- **Core ML (fp16, iOS) is not shipped**: measured in the units that matter for this model — fraction of keypoints landing within 4 px of fp32: median 1.0000 over 10 real images, worst 0.3810.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 21, 512] | 3.353e-06 | 1.000000 |
| 1 | [1, 21, 512] | 2.414e-06 | 1.000000 |

XNNPACK delegate coverage (fp32): 93.9% (306/326 ops); ops left on the portable kernels: `dim_order_ops._to_dim_order_copy.default` x8, `aten.unsqueeze_copy.default` x3, `aten.split_with_sizes_copy.default` x3, `aten.sum.dim_IntList` x2, `aten.pow.Tensor_Scalar` x2, `aten.squeeze_copy.dims` x2

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))

**Notes**: This is a top-down model: it needs a crop around one hand, which means a hand detector upstream. This repository does not ship one — mmpose distributes rtmdet-nano-hand for the job. Pose quality here is verified only as far as the decode contract; judging the keypoints needs hand imagery.
