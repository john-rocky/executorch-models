# edsr_base_x4 — ExecuTorch XNNPACK

`edsr_base_x4_xnnpack_fp32.pte` (6.1 MB, fp32, XNNPACK-delegated)

- **Source**: eugenesiow/edsr-base (super-image)
- **License**: Apache-2.0
- **Input**: [[1, 3, 128, 128]] — RGB 0-1, 128x128 tile
- **Output**: SR image [1,3,512,512] RGB 0-1

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 3, 512, 512] | 3.576e-06 | 1.000000 |

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch 39.6 ms vs torch eager 78.2 ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
