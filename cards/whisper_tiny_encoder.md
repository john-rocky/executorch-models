# whisper_tiny_encoder — ExecuTorch XNNPACK

- **Source**: openai/whisper-tiny
- **License**: Apache-2.0
- **Input**: [[1, 80, 3000]] — log-mel spectrogram [1,80,3000] — 30 s at 16 kHz, 80 mel bins, hop 160, win 400, exactly what WhisperFeatureExtractor produces
- **Output**: encoder_hidden_states [1,1500,384]

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
| fp32 | `whisper_tiny_encoder_xnnpack_fp32.pte` | 32.9 | 1.000000 | 54.5 |
| fp16 | `whisper_tiny_encoder_xnnpack_fp16.pte` | 17.6 | 0.999999 | 110.9 |
| int8 (dynamic) | `whisper_tiny_encoder_xnnpack_int8.pte` | 11.7 | 0.999454 | 59.1 |

\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: 20.5 ms).

### Checked in the task's own units

Correlation is a first filter. These are the numbers that decide:

- **int8 (dynamic)** — measured end to end — decoded token sequences that match fp32: 100% of five decoded sequences are identical when the int8 encoder is swapped in for fp32, decoder held constant.

## Verification (executorch 1.4.0, torch 2.13.0)

Parity is measured against the fp32 eager model on random input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
| 0 | [1, 1500, 384] | 1.836e-04 | 1.000000 |

XNNPACK delegate coverage (fp32): 72.9% (159/218 ops); ops left on the portable kernels: `aten.expand_copy.default` x16, `aten.native_layer_norm.default` x9, `aten.mul.Scalar` x8, `aten.logical_not.default` x8, `aten.eq.Scalar` x4, `aten.full_like.default` x4, `aten.any.dim` x4, `aten.where.self` x4, `aten.arange.start_step` x1, `aten.embedding.default` x1

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models))
