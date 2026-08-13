# dfine_s_coco — ExecuTorch XNNPACK

`dfine_s_coco_xnnpack_fp32.pte` (41.5 MB, fp32, XNNPACK-delegated)

- **Source**: ustc-community/dfine-small-coco
- **License**: Apache-2.0
- **Input**: [[1, 3, 640, 640]] — RGB/255 only (no mean/std norm), 640x640
- **Output**: logits [1,300,80] (sigmoid -> per-class score), boxes [1,300,4] cxcywh normalized 0..1; postprocess = sigmoid + top-k, NO NMS

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 300, 80] | 1.593e-04 | 1.000000 |
| 1 | [1, 300, 4] | 2.003e-05 | 1.000000 |

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch 54.4 ms vs torch eager 138.3 ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
