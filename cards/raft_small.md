# raft_small — ExecuTorch

- **Source**: torchvision raft_small (C_T_V2, FlyingChairs + FlyingThings3D)
- **License**: BSD-3-Clause
- **Input**: [[1, 3, 384, 512], [1, 3, 384, 512]] — two RGB frames, each scaled to [-1,1], 384x512 (both dimensions must stay divisible by 8)
- **Output**: flow [1,2,384,512] in pixels: channel 0 is horizontal displacement from frame 1 to frame 2, channel 1 vertical. Refined over 12 iterations, which are baked into the graph — the intermediate iterates are not returned.

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `raft_small_xnnpack_fp32.pte` | 4.4 | 1.000000 | 169.6 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 135.2 ms).

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on real image input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 2, 384, 512] | 4.792e-04 | 1.000000 |

XNNPACK delegate coverage (fp32): 62.9% (1208/1921 ops); ops left on the portable kernels: `dim_order_ops._to_dim_order_copy.default` x435, `aten.split_with_sizes_copy.default` x50, `aten.grid_sampler_2d.default` x48, `aten.expand_copy.default` x30, `aten.arange.start_step` x28, `aten.lt.Scalar` x24, `aten.sub.Tensor` x24, `aten.where.self` x24, `aten._native_batch_norm_legit.no_stats` x21, `aten.alias_copy.default` x12, `aten.view_copy.default` x6, `aten.avg_pool2d.default` x3, `aten.cat.default` x2, `aten.unsqueeze_copy.default` x2, `aten.repeat.default` x2, `dim_order_ops._clone_dim_order.default` x1, `aten.sqrt.default` x1

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))

**Notes**: No Core ML build: RAFT's correlation volume is a rank-6 tensor and Core ML caps at rank 5. No fp16 build either — this is a conv-only model, where XNNPACK serializes convolution weights as fp32 whatever the graph dtype, so fp16 would be the same size with extra casts.
