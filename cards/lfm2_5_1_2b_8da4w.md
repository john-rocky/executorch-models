# LFM2.5-1.2B-Instruct — ExecuTorch XNNPACK 8da4w

`lfm2_5_1_2b_xnnpack_8da4w.pte` (741 MB)

- **Source**: LiquidAI/LFM2.5-1.2B-Instruct (hybrid conv/attention)
- **License**: LFM Open License v1.0
- **Quantization**: 8da4w (8-bit dynamic activation / 4-bit weight) + 8-bit embedding
  (`embedding_quantize: "8,0"`; cuts 1143 MB → 741 MB vs the fp32-embedding v1)
- **Export**: executorch 1.4.0 `export_llm`, dynamic shape, max_seq_length 2048, XNNPACK extended_ops
- **Config**: `llm_params/lfm2_5_1_2b_xnnpack_8da4w_e8.yaml`

## Verification (2026-08-13)

Mac gate (greedy via `native.py`, chat template): correct 2-sentence Rayleigh-scattering
answer, 170.8 tok/s on M-series Mac (reference only). v1 (fp32 embedding) passed 3/3
(Paris / Japanese / haiku) with identical quant settings otherwise.

iPhone 17 Pro (ETBench, XNNPACK CPU, default threads):

| metric | value |
|--------|-------|
| load | 1.6 s |
| ttft (short prompt) | 0.06-0.07 s |
| decode | **55-81 tok/s** (81 short answer, 55 at 128 tokens) |

Outputs correct (Paris; coherent 128-token story).

**Usage note — chat template is required.** This is an instruct model: raw untemplated
text makes it emit `<|im_end|>` immediately (looks like broken generation but is not).
Always wrap prompts as
`<|startoftext|><|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`, eos ids [7].
