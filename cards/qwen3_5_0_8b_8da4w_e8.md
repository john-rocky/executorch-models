# Qwen3.5-0.8B — ExecuTorch XNNPACK 8da4w + 8-bit embedding

`qwen3_5_0_8b_xnnpack_8da4w_e8.pte` (651.1 MB — down from 1413 MB without embedding
quantization; the 248320x1024 fp32 embedding was ~1 GB of the first export)

- **Source**: Qwen/Qwen3.5-0.8B
- **License**: Apache-2.0
- **Quantization**: 8da4w linear + 8-bit embedding (`embedding_quantize: "8,0"`)
- **Export**: executorch 1.4.0 `export_llm`, static shape (seq_len=1), max_seq_length 2048, XNNPACK extended_ops
- **Config**: `llm_params/qwen3_5_0_8b_xnnpack_8da4w_e8.yaml`

## Verification (Mac arm64, 2026-08-13)

Generation gate 3/3 via `llm_params/gen_static.py` (token-by-token prefill + greedy decode):

| prompt | output | decode tok/s |
|--------|--------|--------------|
| capital of France? | Paris (correct; adds a wrong "second-largest city in Europe" claim — model-level, same as fp32-embedding export) | 21.0 |
| 日本の首都は?(日本語) | 「日本の首都は **東京** です。」 | 21.4 |
| haiku about autumn leaves | 3-line poem | 21.3 |

Decode ~21 tok/s matches the fp32-embedding export (20.6) — the size cut is free.
Prefill tok/s is sequential-prefill reference only (static export).
Chat template: ChatML, bos 248045, eos [248046, 248044].

iPhone 17 Pro (ETBench, XNNPACK CPU, default threads): decode **10.5 tok/s**, ttft 0.58 s,
load 0.7 s — same speed as the 1413 MB fp32-embedding export (9.2-11.1 tok/s), at 46% of its size.
(Thermal note: a hot device throttles to ~7 tok/s; numbers above are from a cool run.)
