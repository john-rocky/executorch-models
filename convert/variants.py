"""Shared view over results/*.json: group a model's precision variants and say
which ones passed their quality gate (= which .pte files may ship)."""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
PRECISIONS = ["fp32", "fp16", "int8"]

# Worst-output corr vs fp32 eager below which a variant is not shippable.
CORR_GATE = {"fp32": 0.999, "fp16": 0.995, "int8": 0.95}
# A reduced precision also has to earn its slot: XNNPACK serializes convolution
# weights as fp32 whatever the graph dtype (op_conv2d passes force_fp32=True), so
# a fp16 CNN is byte-for-byte the fp32 file plus cast ops. Ship only variants that
# come in at least this much smaller.
SIZE_RATIO_MAX = 0.95


def worst_corr(result):
    return min(p["corr"] for p in result["parity"])


def load_variants(name):
    """name -> {precision: result} for every converted variant, each tagged with
    whether it ships and, if not, why ('quality' or 'no_size_gain')."""
    out = {}
    for prec in PRECISIONS:
        suffix = "" if prec == "fp32" else f"_{prec}"
        path = os.path.join(RESULTS, f"{name}{suffix}.json")
        if os.path.exists(path):
            r = json.load(open(path))
            r["worst_corr"] = worst_corr(r)
            r["gate_pass"] = r["worst_corr"] >= CORR_GATE[prec]
            out[prec] = r
    base = out.get("fp32")
    for i, prec in enumerate(PRECISIONS):
        r = out.get(prec)
        if r is None:
            continue
        ratio = r["size_mb"] / base["size_mb"] if base else 1.0
        r["size_ratio"] = ratio
        # Every shipping variant above this one in the list is both more
        # trustworthy and already published, so a variant that is not smaller
        # than one of them is strictly dominated — no reason to pick it.
        dominated = any(out[p]["ships"] and r["size_mb"] > SIZE_RATIO_MAX * out[p]["size_mb"]
                        for p in PRECISIONS[:i] if p in out)
        r["dominated_by"] = next(
            (p for p in PRECISIONS[:i]
             if p in out and out[p]["ships"]
             and r["size_mb"] > SIZE_RATIO_MAX * out[p]["size_mb"]), None)
        # A `quality_override` in the result is a measurement in the task's own
        # units, and it wins. Correlation is a blunt instrument: it cleared
        # LaMa's int8 at 0.958 when the same file is 22 dB PSNR against fp32,
        # which is visible degradation, and it flagged 6DRepNet's int8 at 0.815
        # when the real number was 46 degrees of rotation error. Whenever the
        # output is an image or a small vector, measure the thing that matters
        # and record it here.
        override = r.get("quality_override")
        if override and override.get("verdict") == "fail":
            r["ships"], r["skip_reason"] = False, "task_metric"
        elif override and override.get("verdict") == "pass":
            # A measurement in the task's units overrides the correlation gate in
            # both directions. MoGe's fp16 build reads corr 0.43 purely because
            # one of its four outputs is a near-binary mask whose raw logits
            # correlate badly while the thresholded mask is identical; its point
            # map and normals are cosine 1.000000. Refusing to ship on that would
            # be the same mistake as shipping LaMa's int8 on a passing corr.
            r["ships"] = prec == "fp32" or (ratio <= SIZE_RATIO_MAX and not dominated)
            r["skip_reason"] = None if r["ships"] else (
                "dominated" if dominated else "no_size_gain")
        elif not r["gate_pass"]:
            r["ships"], r["skip_reason"] = False, "quality"
        elif prec == "fp32":
            r["ships"], r["skip_reason"] = True, None
        elif ratio > SIZE_RATIO_MAX:
            r["ships"], r["skip_reason"] = False, "no_size_gain"
        elif dominated:
            r["ships"], r["skip_reason"] = False, "dominated"
        else:
            r["ships"], r["skip_reason"] = True, None
    return out


def model_names():
    """Base model names (fp32 result exists), sorted."""
    names = []
    for f in sorted(os.listdir(RESULTS)):
        if not f.endswith(".json"):
            continue
        stem = f[: -len(".json")]
        if any(stem.endswith(f"_{p}") for p in PRECISIONS[1:]):
            continue
        names.append(stem)
    return names


def label(prec, result):
    if prec == "int8" and result.get("int8_mode", "").startswith("dynamic"):
        return "int8 (dynamic)"
    return prec
