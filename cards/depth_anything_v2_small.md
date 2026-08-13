# depth_anything_v2_small — ExecuTorch XNNPACK

`depth_anything_v2_small_xnnpack_fp32.pte` (99.0 MB, fp32, XNNPACK-delegated)

- **Source**: depth-anything/Depth-Anything-V2-Small-hf
- **License**: Apache-2.0
- **Input**: [[1, 3, 518, 518]] — RGB, ImageNet norm, 518x518
- **Output**: relative inverse depth [1,518,518]

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 518, 518] | 6.914e-06 | 1.000000 |

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch 173.7 ms vs torch eager 88.0 ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
