# clip_vit_b32_image — ExecuTorch XNNPACK

`clip_vit_b32_image_xnnpack_fp32.pte` (351.6 MB, fp32, XNNPACK-delegated)

- **Source**: openai/clip-vit-base-patch32
- **License**: MIT
- **Input**: [[1, 3, 224, 224]] — RGB, CLIP norm (mean .481/.458/.408, std .269/.261/.276), 224x224
- **Output**: image embedding [1,512] (unnormalized; L2-normalize before cosine)

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 512] | 4.768e-06 | 1.000000 |

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch 18.3 ms vs torch eager 17.8 ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
