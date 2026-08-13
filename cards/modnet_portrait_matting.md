# modnet_portrait_matting — ExecuTorch XNNPACK

`modnet_portrait_matting_xnnpack_fp32.pte` (26.1 MB, fp32, XNNPACK-delegated)

- **Source**: ZHKKKe/MODNet + DavG25/modnet-pretrained-models ckpt
- **License**: Apache-2.0
- **Input**: [[1, 3, 512, 512]] — RGB [-1,1], 512x512
- **Output**: alpha matte [1,1,512,512] 0-1

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 1, 512, 512] | 1.572e-04 | 1.000000 |

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch 69.5 ms vs torch eager 120.2 ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
