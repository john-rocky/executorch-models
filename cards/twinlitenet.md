# twinlitenet — ExecuTorch XNNPACK

- **Source**: chequanghuy/TwinLiteNet (pretrained/best.pth)
- **License**: MIT
- **Input**: [[1, 3, 360, 640]] — RGB 0-1, 360x640
- **Output**: drivable area [1,2,360,640] + lane line [1,2,360,640]

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `twinlitenet_xnnpack_fp32.pte` | 1.8 | 1.000000 | 32.5 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 23.9 ms).

### Precisions that did not earn a slot

- **fp16 is not shipped**: it comes out at 100% of the fp32 file (1.8 MB vs 1.8 MB), so it buys nothing. XNNPACK serializes convolution weights as fp32 no matter what dtype the graph carries, so on a conv-heavy model fp16 saves no disk and only adds cast operations. Reach for int8 here, not fp16.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 2, 360, 640] | 1.240e-05 | 1.000000 |
| 1 | [1, 2, 360, 640] | 1.287e-05 | 1.000000 |

XNNPACK delegate coverage (fp32): 79.7% (177/222 ops); ops left on the portable kernels: `aten.gt.Scalar` x20, `aten.where.self` x20, `aten.avg_pool2d.default` x3, `aten.max.dim` x1, `aten.expand_copy.default` x1

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))

**Notes**: int8 does not run on executorch 1.4.0, and the reason is a collision between two workarounds rather than anything about this model. PReLU segfaults on the XNNPACK delegate (upstream #17559, fix #21480 not in the stable wheel), so it has to be excluded from partitioning — and a quantized graph with a portable PReLU between delegated convolutions fails shape propagation at execute. Three conv+PReLU layers reproduce it; the same graph in fp32 is corr 1.000000. Reported at https://github.com/pytorch/executorch/pull/21480
