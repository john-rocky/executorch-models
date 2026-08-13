"""Render results/*.json into cards/<name>.md (HF-ready model cards, English)."""
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARDS = os.path.join(REPO, "cards")
os.makedirs(CARDS, exist_ok=True)

TEMPLATE = """# {name} — ExecuTorch XNNPACK

`{pte}` ({size_mb} MB, fp32, XNNPACK-delegated)

- **Source**: {source}
- **License**: {license}
- **Input**: {inputs} — {preprocess}
- **Output**: {outputs}

## Verification (Mac arm64, executorch {executorch}, torch {torch})

Parity vs torch fp32 eager on random input:

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
{parity_rows}

Median latency over 10 runs (single Mac process, reference only — device numbers to follow):
ExecuTorch {et_ms_median} ms vs torch eager {torch_eager_ms_median} ms.

## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: `convert/` in the conversion repo){notes}
"""

for path in sorted(glob.glob(os.path.join(REPO, "results", "*.json"))):
    r = json.load(open(path))
    rows = "\n".join(
        f"| {p['output']} | {p['shape']} | {p['max_abs_diff']:.3e} | {p['corr']:.6f} |"
        for p in r["parity"]
    )
    notes = ""
    if "notes" in r:
        notes = "\n\n**Notes**: " + r["notes"]
    card = TEMPLATE.format(parity_rows=rows, notes=notes, **{
        k: r.get(k, "?") for k in
        ["name", "pte", "size_mb", "source", "license", "inputs", "preprocess",
         "outputs", "executorch", "torch", "et_ms_median", "torch_eager_ms_median"]
    })
    out = os.path.join(CARDS, f"{r['name']}.md")
    with open(out, "w") as f:
        f.write(card)
    print("wrote", out)
