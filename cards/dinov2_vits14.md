# dinov2_vits14 — ExecuTorch

- **Source**: facebookresearch/dinov2 (torch.hub)
- **License**: Apache-2.0
- **Input**: [[1, 3, 518, 518]] — RGB, ImageNet norm, 518x518
- **Output**: cls token [1,384], patch tokens [1,1369,384]

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `dinov2_vits14_xnnpack_fp32.pte` | 88.4 | 1.000000 | 159.0 |
| fp16 | `dinov2_vits14_xnnpack_fp16.pte` | 44.8 | 0.999945 | 283.7 |
| int8 (dynamic) | `dinov2_vits14_xnnpack_int8.pte` | 24.9 | 0.998009 | 155.7 |
| Core ML (fp16, iOS) | `dinov2_vits14_coreml_all.pte` | 44.7 | 0.999863 | 41.2 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. On an iPhone 17 Pro, Depth-Anything-V2-Small runs
500.8 ms through XNNPACK and 42.7 ms through Core ML, at half the file size. It
computes in fp16 and is iOS-only; the XNNPACK files stay the portable option and
are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 50.8 ms).

### Checked in the task's own units

Correlation is a first filter. These are the numbers that decide:

- **int8 (dynamic)** — measured in the units that matter for this model — cosine similarity of the embeddings: median 0.9986 over 10 real images, worst 0.9965.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 384] | 3.219e-05 | 1.000000 |
| 1 | [1, 1369, 384] | 4.311e-04 | 1.000000 |

XNNPACK delegate coverage (fp32): 66.7% (414/621 ops); ops left on the portable kernels: `aten.expand_copy.default` x49, `aten.squeeze_copy.dims` x36, `aten.native_layer_norm.default` x25, `aten.mul.Scalar` x24, `aten.logical_not.default` x24, `aten.eq.Scalar` x12, `aten.full_like.default` x12, `aten.any.dim` x12, `aten.where.self` x12, `aten.select_copy.int` x1

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
