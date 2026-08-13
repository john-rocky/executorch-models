# rtdetrv2_s_r18vd — ExecuTorch XNNPACK

`rtdetrv2_s_r18vd_xnnpack_fp32.pte` (80.9 MB, fp32, XNNPACK-delegated)

- **Source**: PekingU/rtdetr_v2_r18vd
- **License**: Apache-2.0
- **Input**: [[1, 3, 640, 640]] — RGB/255 only (no mean/std norm), 640x640
- **Output**: logits [1,300,80] (sigmoid -> per-class score), boxes [1,300,4] cxcywh normalized 0..1; postprocess = sigmoid + top-k, NO NMS

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 300, 80] | 3.958e-05 | 1.000000 |
| 1 | [1, 300, 4] | 1.138e-05 | 1.000000 |

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch 55.5 ms vs torch eager 64.8 ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
