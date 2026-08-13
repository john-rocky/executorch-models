# yolox_s — ExecuTorch XNNPACK

`yolox_s_xnnpack_fp32.pte` (35.9 MB, fp32, XNNPACK-delegated)

- **Source**: Megvii-BaseDetection/YOLOX (yolox_s)
- **License**: Apache-2.0
- **Input**: [[1, 3, 640, 640]] — BGR 0..255 float, NO normalization (YOLOX v0.3+ convention), 640x640 letterbox pad 114
- **Output**: [1,8400,85]: cx,cy,w,h (input px), objectness, 80 class scores; postprocess = obj*cls threshold + NMS (required)

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 8400, 85] | 3.021e-03 | 1.000000 |

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch 24.7 ms vs torch eager 38.8 ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))

**Notes**: max_abs_diff ~3e-3 is on decoded pixel-coordinate outputs (values up to 640), i.e. relative error <1e-5; corr 1.000000.
