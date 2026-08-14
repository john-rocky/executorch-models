"""Render results/*.json into cards/<name>.md (HF-ready model cards, English).

Precision variants are grouped into one card: a comparison table of the variants
that passed their quality gate, plus an honest note about any that did not.
"""
import os

from variants import CORR_GATE, PRECISIONS, label, load_variants, model_names

SKIP_TEXT = {
    "quality": (
        "- **{lab} is not shipped**: worst-output corr {corr:.3f} against fp32 eager, "
        "below the {gate} bar for this precision. The file converts and runs; the "
        "numbers do not hold up, so it is left out rather than shipped with a warning."),
    "no_size_gain": (
        "- **{lab} is not shipped**: it comes out at {pct:.0f}% of the fp32 file "
        "({size} MB vs {base} MB), so it buys nothing. XNNPACK serializes convolution "
        "weights as fp32 no matter what dtype the graph carries, so on a conv-heavy "
        "model fp16 saves no disk and only adds cast operations. Reach for int8 here, "
        "not fp16."),
    "task_metric": (
        "- **{lab} is not shipped**: {why}"),
    "dominated": (
        "- **{lab} is not shipped**: at {size} MB it is no smaller than the {dom} "
        "build, which is also the more faithful of the two. Nothing would pick it."),
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARDS = os.path.join(REPO, "cards")
os.makedirs(CARDS, exist_ok=True)

TEMPLATE = """# {name} — ExecuTorch XNNPACK

- **Source**: {source}
- **License**: {license}
- **Input**: {inputs} — {preprocess}
- **Output**: {outputs}

## Variants

All variants take and return fp32 tensors — swap the `.pte` file, keep your app code.

| precision | file | size (MB) | parity vs fp32 eager (worst corr) | Mac median (ms)* |
|-----------|------|-----------|------------------------------------|------------------|
{variant_rows}

\\*Mac arm64, single process, median of 10 — a reference point for relative cost
only, not a device number (torch eager fp32 on the same machine: {torch_eager_ms_median} ms).
{failed}
## Verification (executorch {executorch}, torch {torch})

Parity is measured against the fp32 eager model on {gate_input} input; `corr` is
the correlation over all elements of each output tensor.

| output | shape | max_abs_diff | corr |
|--------|-------|--------------|------|
{parity_rows}
{delegation}
## Conversion

torch.export -> to_edge_transform_and_lower(XnnpackPartitioner) -> .pte
(conversion scripts: [executorch-models](https://github.com/john-rocky/executorch-models)){notes}
"""


def main():
    for name in model_names():
        variants = load_variants(name)
        r = variants["fp32"]
        vrows, failed_lines, audited = [], [], []
        for prec in PRECISIONS:
            v = variants.get(prec)
            if not v:
                continue
            if v["ships"]:
                ov = v.get("quality_override")
                corr_cell = f"{v['worst_corr']:.6f}"
                if ov and ov.get("verdict") == "pass" and v["worst_corr"] < CORR_GATE[prec]:
                    # Do not leave a misleading number standing alone in the table.
                    corr_cell += " — see below"
                vrows.append(f"| {label(prec, v)} | `{v['pte']}` | {v['size_mb']} | "
                             f"{corr_cell} | {v['et_ms_median']} |")
                if ov and ov.get("verdict") == "pass":
                    audited.append(f"- **{label(prec, v)}** — {ov['why']}")
            else:
                dom = v.get("dominated_by")
                failed_lines.append(SKIP_TEXT[v["skip_reason"]].format(
                    lab=label(prec, v), corr=v["worst_corr"], gate=CORR_GATE[prec],
                    pct=100 * v["size_ratio"], size=v["size_mb"],
                    base=variants["fp32"]["size_mb"],
                    dom=label(dom, variants[dom]) if dom else "smaller",
                    why=(v.get("quality_override") or {}).get("why", "")))
        failed = ""
        if audited:
            failed += ("\n### Checked in the task's own units\n\n"
                       "Correlation is a first filter. These are the numbers that decide:\n\n"
                       + "\n".join(audited) + "\n")
        if failed_lines:
            failed += ("\n### Precisions that did not earn a slot\n\n"
                       + "\n".join(failed_lines) + "\n")
        prows = "\n".join(
            f"| {p['output']} | {p['shape']} | {p['max_abs_diff']:.3e} | {p['corr']:.6f} |"
            for p in r["parity"])
        delegation = ""
        d = r.get("delegation")
        if d:
            total = d["delegated_ops"] + d["portable_ops"]
            delegation = (f"\nXNNPACK delegate coverage (fp32): {d['coverage_pct']}% "
                          f"({d['delegated_ops']}/{total} ops)")
            if d["portable_fallback"]:
                ops = ", ".join(f"`{k}` x{v}" for k, v in d["portable_fallback"].items())
                delegation += f"; ops left on the portable kernels: {ops}"
            delegation += "\n"
        notes = ""
        for prec in PRECISIONS:
            v = variants.get(prec)
            if v and v.get("notes"):
                tag = "" if prec == "fp32" else f" ({prec})"
                notes += f"\n\n**Notes{tag}**: {v['notes']}"
        card = TEMPLATE.format(
            variant_rows="\n".join(vrows), parity_rows=prows, delegation=delegation,
            failed=failed, notes=notes,
            **{k: r.get(k, "?") for k in
               ["name", "source", "license", "inputs", "preprocess", "outputs",
                "executorch", "torch", "torch_eager_ms_median"]},
            gate_input=r.get("gate_input", "random"))
        out = os.path.join(CARDS, f"{r['name']}.md")
        with open(out, "w") as f:
            f.write(card)
        print("wrote", out)


if __name__ == "__main__":
    main()
