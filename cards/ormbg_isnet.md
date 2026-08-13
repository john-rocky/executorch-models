# ormbg_isnet — ExecuTorch XNNPACK

`ormbg_isnet_xnnpack_fp32.pte` (176.1 MB, fp32, XNNPACK-delegated)

- **Source**: schirrmacher/ormbg
- **License**: Apache-2.0
- **Input**: [[1, 3, 1024, 1024]] — RGB 0-1, 1024x1024
- **Output**: alpha mask [1,1,1024,1024] 0-1 (sigmoid)

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 1, 1024, 1024] | 2.831e-06 | 1.000000 |

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch 123.8 ms vs torch eager 387.3 ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
