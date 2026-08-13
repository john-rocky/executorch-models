# Minimal generation harness for static-shape (seq_len=1) ExecuTorch LLM exports.
# Prefills token-by-token, then greedy-decodes. Reports prefill tok/s and decode tok/s.
import argparse
import json
import time

import torch

from executorch.extension.pybindings.portable_lib import _load_for_executorch
from executorch.extension.pybindings import portable_lib  # noqa
from executorch.extension.llm.custom_ops import custom_ops  # noqa
from executorch.kernels import quantized  # noqa

from pytorch_tokenizers import get_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pte", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--tokenizer_config", default=None)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max_new", type=int, default=96)
    p.add_argument("--eos_ids", type=str, default="[]", help="JSON list of extra EOS token ids")
    args = p.parse_args()

    tokenizer = get_tokenizer(args.tokenizer, args.tokenizer_config)
    eos_ids = set(json.loads(args.eos_ids))
    if hasattr(tokenizer, "stop_tokens"):
        eos_ids.update(tokenizer.stop_tokens)
    eos_ids.add(tokenizer.eos_id)

    model = _load_for_executorch(args.pte)

    prompt_tokens = tokenizer.encode(args.prompt, bos=0, eos=0)
    print(f"prompt tokens: {len(prompt_tokens)}")

    # Token-by-token prefill.
    t0 = time.perf_counter()
    logits = None
    for i, tok in enumerate(prompt_tokens):
        logits = model.forward(
            (
                torch.tensor([[tok]], dtype=torch.long),
                torch.tensor([i], dtype=torch.long),
            )
        )[0]
    prefill_time = time.perf_counter() - t0
    print(f"prefill: {len(prompt_tokens)} tok in {prefill_time:.2f}s "
          f"({len(prompt_tokens)/prefill_time:.2f} tok/s)")

    # Greedy decode.
    pos = len(prompt_tokens)
    out_tokens = []
    t1 = time.perf_counter()
    cur = int(torch.argmax(logits[:, -1, :] if logits.dim() == 3 else logits, dim=-1).item())
    while len(out_tokens) < args.max_new:
        if cur in eos_ids:
            break
        out_tokens.append(cur)
        logits = model.forward(
            (
                torch.tensor([[cur]], dtype=torch.long),
                torch.tensor([pos], dtype=torch.long),
            )
        )[0]
        pos += 1
        cur = int(torch.argmax(logits[:, -1, :] if logits.dim() == 3 else logits, dim=-1).item())
    decode_time = time.perf_counter() - t1

    text = tokenizer.decode(out_tokens)
    print("=== OUTPUT ===")
    print(text)
    print("==============")
    if out_tokens:
        print(f"decode: {len(out_tokens)} tok in {decode_time:.2f}s "
              f"({len(out_tokens)/decode_time:.2f} tok/s)")


if __name__ == "__main__":
    main()
