# efficientnet_b1 — ExecuTorch XNNPACK

`efficientnet_b1_xnnpack_fp32.pte` (31.2 MB, fp32, XNNPACK-delegated)

- **Source**: torchvision efficientnet_b1 IMAGENET1K_V2
- **License**: BSD-3-Clause
- **Input**: [[1, 3, 240, 240]] — RGB, ImageNet norm, 240x240
- **Output**: ImageNet logits [1,1000]

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 1000] | 6.676e-06 | 1.000000 |

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch 9.2 ms vs torch eager 345.4 ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
