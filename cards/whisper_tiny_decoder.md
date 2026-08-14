# whisper_tiny_decoder — ExecuTorch

- **Source**: openai/whisper-tiny
- **License**: Apache-2.0
- **Input**: [[1, 1500, 384], [1, 128]] — encoder_hidden_states [1,1500,384] + decoder_input_ids [1,128] int64, left-aligned and padded; start with [<|startoftranscript|>, <|lang|>, <|transcribe|>, <|notimestamps|>]
- **Output**: logits [1,128,51865]; greedy step = argmax of row (len-1), append, re-run; stop at <|endoftext|> (50257)

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| build | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `whisper_tiny_decoder_xnnpack_fp32.pte` | 198.0 | 1.000000 | 18.1 |
| fp16 | `whisper_tiny_decoder_xnnpack_fp16.pte` | 99.1 | 0.999988 | 48.8 |
| Core ML (fp16, iOS) | `whisper_tiny_decoder_coreml_all.pte` | 59.3 | 0.999910 | 3.1 |


The Core ML build is the same graph lowered to Apple's Neural Engine instead of
XNNPACK, which is CPU-only. On an iPhone 17 Pro, Depth-Anything-V2-Small runs
500.8 ms through XNNPACK and 42.7 ms through Core ML, at half the file size. It
computes in fp16 and is iOS-only; the XNNPACK files stay the portable option and
are what runs on Android.

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 12.1 ms).

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on random input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 128, 51865] | 1.183e-04 | 1.000000 |

XNNPACK delegate coverage (fp32): 64.6% (287/444 ops); ops left on the portable kernels: `aten.expand_copy.default` x33, `aten.mul.Scalar` x16, `aten.logical_not.default` x16, `aten.native_layer_norm.default` x13, `aten.where.self` x12, `aten.unsqueeze_copy.default` x10, `aten.scalar_tensor.default` x8, `aten.eq.Scalar` x8, `aten.full_like.default` x8, `aten.any.dim` x8, `aten.arange.start_step` x4, `aten.add.Tensor` x3, `aten.index.Tensor` x3, `aten.slice_copy.Tensor` x3, `aten.sub.Tensor` x2, `aten.bitwise_and.Tensor` x2, `aten.full.default` x1, `aten.embedding.default` x1, `aten.repeat.default` x1, `aten.le.Tensor` x1, `aten.cat.default` x1, `aten.ne.Scalar` x1, `aten.cumsum.default` x1, `aten.eq.Tensor` x1

## Conversion

torch.export -> to_edge_transform_and_lower(partitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
