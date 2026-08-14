# Whisper-tiny — ExecuTorch XNNPACK (encoder + decoder)

Speech recognition in two `.pte` files: the encoder runs once per 30-second window,
the decoder once per generated token.

| graph | precision | file | size (MB) | corr vs fp32 eager |
|-------|-----------|------|-----------|--------------------|
| encoder | fp32 | `whisper_tiny_encoder_xnnpack_fp32.pte` | 32.9 | 1.000000 |
| encoder | fp16 | `whisper_tiny_encoder_xnnpack_fp16.pte` | 17.6 | 0.999999 |
| encoder | int8 | `whisper_tiny_encoder_xnnpack_int8.pte` | 11.7 | 0.999454 |
| decoder | fp32 | `whisper_tiny_decoder_xnnpack_fp32.pte` | 198.0 | 1.000000 |
| decoder | fp16 | `whisper_tiny_decoder_xnnpack_fp16.pte` | 99.1 | 0.999988 |

Every file takes and returns fp32 tensors (token ids stay int64), so any encoder
pairs with any decoder. The lightest working pair is 110.8 MB.

- **Source**: [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)
- **License**: Apache-2.0
- **Encoder input**: log-mel spectrogram `[1,80,3000]` — 30 s at 16 kHz, 80 mel bins,
  hop 160, window 400. This is exactly what `WhisperFeatureExtractor` produces; pad
  or trim audio to 30 s as it does.
- **Encoder output**: `encoder_hidden_states [1,1500,384]`
- **Decoder input**: the encoder output plus `decoder_input_ids [1,128]` int64,
  left-aligned and padded. Start the sequence with
  `<|startoftranscript|>`, a language token, `<|transcribe|>`, `<|notimestamps|>`.
- **Decoder output**: `logits [1,128,51865]`

## Decoding

There is no KV cache. The decoder is a static graph over a fixed 128-token window,
so a greedy step is: take `argmax` of row `len-1`, append it, run again. Stop at
`<|endoftext|>` (50257). 128 tokens covers a 30-second window of ordinary speech
with room to spare; for longer audio, start a new window.

That costs a full 128-position forward pass per token. On a 37M-parameter model
this is cheap enough to be practical, and it keeps the graph static — which is what
lets the same file run unchanged across runtimes and precisions.

## Verification (Mac arm64, executorch 1.4.0, torch 2.13.0)

The two wrappers compose back to `WhisperForConditionalGeneration` exactly
(max_abs_diff 0.000e+00), and every graph matches torch fp32 eager at the
correlations in the table above.

Median over 5 runs, Mac arm64 single process — a relative reference, not a device
number: encoder 54.5 ms (torch eager 20.5 ms), decoder 18.1 ms (eager 12.1 ms).

## Two things worth knowing about the sizes

**The decoder .pte is larger than the decoder's weights.** Its parameters come to
118 MB, and the file is 198 MB. Whisper ties `proj_out.weight` to
`decoder.embed_tokens.weight` — one 19.9M-parameter tensor — but the two uses need
different representations in the `.pte`: an embedding table the portable kernels
index into, and the same values packed into the XNNPACK delegate's blob for the
output matmul. Tying them in PyTorch does not tie them here, and referencing the
embedding weight directly through `F.linear` does not either.

**The decoder has no int8 build.** PT2E puts an observer on the int64
`decoder_input_ids` feeding the token embedding, and the lookup then refuses a float
index (`tensors used as indices must be long, int, byte or bool`). The encoder takes
float mel input and quantizes without complaint, which is where the size is worth
taking anyway.

## Conversion

torch.export → to_edge_transform_and_lower(XnnpackPartitioner) → .pte
(conversion script: [executorch-models](https://github.com/john-rocky/executorch-models))

The ExecuTorch tree ships a single-graph Whisper example under
`examples/models/whisper`. This is that model with the halves separated, because a
combined graph re-encodes the audio on every decoded token.
