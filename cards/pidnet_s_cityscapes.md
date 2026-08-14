# pidnet_s_cityscapes — ExecuTorch XNNPACK

- **Source**: XuJiacong/PIDNet + oenpu/PIDNet_S_enlight_friendly_onnx weights
- **License**: MIT
- **Input**: [[1, 3, 1024, 1024]] — RGB, ImageNet norm, 1024x1024
- **Output**: class logits [1,19,128,128] (argmax + upsample in app)

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `pidnet_s_cityscapes_xnnpack_fp32.pte` | 30.5 | 1.000000 | 27.6 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 64.5 ms).

### Precisions that did not earn a slot

- **fp16 is not shipped**: it comes out at 100% of the fp32 file (30.5 MB vs 30.5 MB), so it buys nothing. XNNPACK serializes convolution weights as fp32 no matter what dtype the graph carries, so on a conv-heavy model fp16 saves no disk and only adds cast operations. Reach for int8 here, not fp16.
- **int8 is not shipped**: measured in the units that matter for this model — fraction of pixels keeping their class: median 0.9802 over 10 real images, worst 0.8997.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 19, 128, 128] | 1.696e-05 | 1.000000 |

XNNPACK delegate coverage (fp32): 96.3% (263/273 ops); ops left on the portable kernels: `dim_order_ops._to_dim_order_copy.default` x3, `aten.avg_pool2d.default` x3, `aten.sum.dim_IntList` x2, `aten.unsqueeze_copy.default` x2

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
