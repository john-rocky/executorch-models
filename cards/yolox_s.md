# yolox_s — ExecuTorch XNNPACK

- **Source**: Megvii-BaseDetection/YOLOX (yolox_s)
- **License**: Apache-2.0
- **Input**: [[1, 3, 640, 640]] — BGR 0..255 float, NO normalization (YOLOX v0.3+ convention), 640x640 letterbox pad 114
- **Output**: [1,8400,85]: cx,cy,w,h (input px), objectness, 80 class scores; postprocess = obj*cls threshold + NMS (required)

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `yolox_s_xnnpack_fp32.pte` | 35.9 | 1.000000 | 25.0 |
| int8 | `yolox_s_xnnpack_int8.pte` | 9.2 | 0.998843 | 33.2 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 38.6 ms).

### Precisions that did not earn a slot

- **fp16 is not shipped**: it comes out at 101% of the fp32 file (36.1 MB vs 35.9 MB), so it buys nothing. XNNPACK serializes convolution weights as fp32 no matter what dtype the graph carries, so on a conv-heavy model fp16 saves no disk and only adds cast operations. Reach for int8 here, not fp16.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 8400, 85] | 2.258e-03 | 1.000000 |

XNNPACK delegate coverage (fp32): 88.2% (351/398 ops); ops left on the portable kernels: `aten.view_copy.default` x9, `aten.slice_copy.Tensor` x8, `aten.arange.start_step` x6, `aten.expand_copy.default` x6, `aten.unsqueeze_copy.default` x6, `aten.cat.default` x5, `aten.full.default` x3, `dim_order_ops._to_dim_order_copy.default` x2, `aten.upsample_nearest2d.vec` x2

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
