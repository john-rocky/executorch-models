# ssdlite320_mobilenetv3 — ExecuTorch XNNPACK

`ssdlite320_mobilenetv3_xnnpack_fp32.pte` (13.8 MB, fp32, XNNPACK-delegated)

- **Source**: torchvision ssdlite320_mobilenet_v3_large COCO_V1
- **License**: BSD-3-Clause
- **Input**: [[1, 3, 320, 320]] — RGB 0-1, 320x320 (torchvision SSDLite norm baked in model)
- **Output**: 12 raw heads: (cls [1,A*91,H,W], box [1,A*4,H,W]) x 6 levels, H=W in {20,10,5,3,2,1}

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 546, 20, 20] | 2.241e-05 | 1.000000 |
| 1 | [1, 24, 20, 20] | 3.958e-05 | 1.000000 |
| 2 | [1, 546, 10, 10] | 1.705e-05 | 1.000000 |
| 3 | [1, 24, 10, 10] | 4.649e-06 | 1.000000 |
| 4 | [1, 546, 5, 5] | 2.158e-05 | 1.000000 |
| 5 | [1, 24, 5, 5] | 7.510e-06 | 1.000000 |
| 6 | [1, 546, 3, 3] | 2.438e-05 | 1.000000 |
| 7 | [1, 24, 3, 3] | 7.123e-06 | 1.000000 |
| 8 | [1, 546, 2, 2] | 1.967e-05 | 1.000000 |
| 9 | [1, 24, 2, 2] | 4.768e-06 | 1.000000 |
| 10 | [1, 546, 1, 1] | 2.933e-05 | 1.000000 |
| 11 | [1, 24, 1, 1] | 2.146e-06 | 1.000000 |

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch 5.1 ms vs torch eager 117.3 ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
