# clip_vit_b32_text — ExecuTorch XNNPACK

`clip_vit_b32_text_xnnpack_fp32.pte` (253.9 MB, fp32, XNNPACK-delegated)

- **Source**: openai/clip-vit-base-patch32
- **License**: MIT
- **Input**: [[1, 77], [1, 77]] — CLIP BPE tokens, fixed len 77 with attention mask
- **Output**: text embedding [1,512] (unnormalized; L2-normalize before cosine)

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 512] | 1.907e-06 | 1.000000 |

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch 18.2 ms vs torch eager 16.1 ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
