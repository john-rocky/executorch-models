"""Whisper-tiny (multilingual ASR) -> 2x ExecuTorch XNNPACK .pte.

Split the same way as the SAM family, and for the same reason: the encoder runs
once per 30-second window while the decoder runs once per generated token, so
putting them in one graph would re-encode the audio on every step. The
ExecuTorch tree ships a single-graph Whisper example (examples/models/whisper);
this is that model with the two halves separated.

  encoder: log-mel (1,80,3000) -> encoder_hidden_states (1,1500,384)
  decoder: (encoder_hidden_states, decoder_input_ids (1,128) int64)
           -> logits (1,128,51865)

No KV cache: the decoder is a plain static graph over a fixed 128-token window,
so a step of greedy decoding is "append the token you just took, run again, read
row len-1". 128 covers a 30-second window of ordinary speech with room to spare;
past that, start a new window.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from harness import convert_and_gate
from transformers import WhisperForConditionalGeneration

CKPT = "openai/whisper-tiny"
MAX_TOKENS = 128
model = WhisperForConditionalGeneration.from_pretrained(CKPT).eval()


class Encoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.enc = m.model.encoder

    def forward(self, input_features):
        return self.enc(input_features).last_hidden_state.contiguous()


class Decoder(nn.Module):
    """Whisper ties `proj_out.weight` to `decoder.embed_tokens.weight`, but
    holding both modules gives the exported graph two constant paths to the same
    19.9M-parameter tensor and it lands in the .pte twice — 198 MB for a decoder
    whose weights are 118 MB. Referencing the embedding weight directly through
    F.linear leaves one path and one copy."""

    def __init__(self, m):
        super().__init__()
        self.dec = m.model.decoder

    def forward(self, encoder_hidden_states, decoder_input_ids):
        h = self.dec(input_ids=decoder_input_ids,
                     encoder_hidden_states=encoder_hidden_states,
                     use_cache=False).last_hidden_state
        return F.linear(h, self.dec.embed_tokens.weight)


mel = torch.randn(1, 80, 3000)
tokens = torch.full((1, MAX_TOKENS), model.config.decoder_start_token_id, dtype=torch.long)

enc = Encoder(model).eval()
dec = Decoder(model).eval()

with torch.no_grad():
    hidden = enc(mel)
    logits = dec(hidden, tokens)
    ref = model(input_features=mel, decoder_input_ids=tokens, use_cache=False).logits
comp = (logits - ref).abs().max().item()
print(f"composition max_abs_diff vs WhisperForConditionalGeneration: {comp:.3e}")
assert comp < 1e-3, "encoder/decoder split diverges from the full model"

for prec in (sys.argv[1:] or ["fp32", "fp16", "int8"]):
    convert_and_gate(
        "whisper_tiny_encoder", enc, (mel,), runs=5, precision=prec,
        int8_dynamic=True,  # transformer: static int8 wrecks attention
        extra_meta={
            "source": CKPT,
            "license": "Apache-2.0",
            "preprocess": "log-mel spectrogram [1,80,3000] — 30 s at 16 kHz, 80 mel "
                          "bins, hop 160, win 400, exactly what WhisperFeatureExtractor "
                          "produces",
            "outputs": "encoder_hidden_states [1,1500,384]",
        },
    )
    if prec == "int8":
        # PT2E puts an observer on the int64 decoder_input_ids feeding the token
        # embedding, and the lookup then rejects a float index ("tensors used as
        # indices must be long, int, byte or bool"). The encoder takes float mel
        # and quantizes fine, so int8 is available where it matters.
        print("skip decoder int8: PT2E observes the int64 token ids")
        continue
    convert_and_gate(
        "whisper_tiny_decoder", dec, (hidden, tokens), runs=5, precision=prec,
        int8_dynamic=True,
        extra_meta={
            "source": CKPT,
            "license": "Apache-2.0",
            "preprocess": f"encoder_hidden_states [1,1500,384] + decoder_input_ids "
                          f"[1,{MAX_TOKENS}] int64, left-aligned and padded; start with "
                          f"[<|startoftranscript|>, <|lang|>, <|transcribe|>, "
                          f"<|notimestamps|>]",
            "outputs": f"logits [1,{MAX_TOKENS},51865]; greedy step = argmax of row "
                       f"(len-1), append, re-run; stop at <|endoftext|> (50257)",
        },
    )
