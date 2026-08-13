# twinlitenet — ExecuTorch XNNPACK

`twinlitenet_xnnpack_fp32.pte` (1.8 MB, fp32, XNNPACK-delegated)

- **Source**: chequanghuy/TwinLiteNet (pretrained/best.pth)
- **License**: MIT
- **Input**: [[1, 3, 360, 640]] — RGB 0-1, 360x640
- **Output**: drivable area [1,2,360,640] + lane line [1,2,360,640]

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 2, 360, 640] | 1.192e-05 | 1.000000 |
| 1 | [1, 2, 360, 640] | 1.049e-05 | 1.000000 |

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch 33.7 ms vs torch eager 23.3 ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))

**Notes**: PReLU is excluded from XNNPACK delegation (XNNPACK PReLU segfaults at execute on macOS arm64 in executorch 1.4.0; minimal repro: a lone nn.PReLU(1)). PReLU runs on the portable kernel instead; outputs are bit-identical to the stock model.
