# pidnet_s_cityscapes — ExecuTorch XNNPACK

`pidnet_s_cityscapes_xnnpack_fp32.pte` (30.5 MB, fp32, XNNPACK-delegated)

- **Source**: XuJiacong/PIDNet + oenpu/PIDNet_S_enlight_friendly_onnx weights
- **License**: MIT
- **Input**: [[1, 3, 1024, 1024]] — RGB, ImageNet norm, 1024x1024
- **Output**: class logits [1,19,128,128] (argmax + upsample in app)

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 19, 128, 128] | 1.025e-05 | 1.000000 |

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch 27.6 ms vs torch eager 63.5 ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
