# dinov2_vits14 — ExecuTorch XNNPACK

`dinov2_vits14_xnnpack_fp32.pte` (88.4 MB, fp32, XNNPACK-delegated)

- **Source**: facebookresearch/dinov2 (torch.hub)
- **License**: Apache-2.0
- **Input**: [[1, 3, 518, 518]] — RGB, ImageNet norm, 518x518
- **Output**: cls token [1,384], patch tokens [1,1369,384]

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 384] | 2.494e-05 | 1.000000 |
| 1 | [1, 1369, 384] | 1.180e-04 | 1.000000 |

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch 161.2 ms vs torch eager 50.1 ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
