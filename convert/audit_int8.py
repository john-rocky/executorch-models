"""Audit shipped int8 builds with each model's own metric, not correlation.

Correlation misses in both directions: it cleared LaMa's int8 at 0.958 when the
file is 22 dB PSNR against fp32, and it flagged 6DRepNet's at 0.815 when the real
number was 46 degrees. Everything already published deserves the same check in the
units that matter — masks in IoU, detections in box agreement, images in PSNR.

Writes the result into `results/<name>_int8.json` as `quality_override`, which
`variants.py` honours over the correlation gate and `gen_cards.py` prints.

Usage: python convert/audit_int8.py [name ...]
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from calib import calib_loader
from executorch.runtime import Runtime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, prec):
    p = os.path.join(REPO, "pte", f"{name}_xnnpack_{prec}.pte")
    return Runtime.get().load_program(p).load_method("forward")


def psnr(a, b):
    a, b = a.clamp(0, 1), b.clamp(0, 1)
    return 10 * math.log10(1.0 / max(((a - b) ** 2).mean().item(), 1e-12))


def mask_iou(a, b, thr=0.5):
    x, y = a > thr, b > thr
    union = (x | y).sum().item()
    return (x & y).sum().item() / union if union else 1.0


def label_agreement(a, b):
    """Fraction of pixels whose argmax class is unchanged."""
    return (a.argmax(1) == b.argmax(1)).float().mean().item()


# name -> (calib category, size, norm, metric, unit, pass threshold, extra kwargs)
AUDITS = {
    "modnet_portrait_matting": ("portrait", 512, "pm1", "iou", "mask IoU at 0.5", 0.95, {}),
    "ormbg_isnet": ("portrait", 1024, "01", "iou", "mask IoU at 0.5", 0.95, {}),
    "u2net": ("general", 320, "imagenet", "iou", "mask IoU at 0.5", 0.95, {}),
    "dis_isnet": ("general", 1024, "pm1", "iou", "mask IoU at 0.5", 0.95, {}),
    "pidnet_s_cityscapes": ("street", 1024, "imagenet", "labels",
                            "fraction of pixels keeping their class", 0.95, {}),
    "edsr_base_x4": ("general", 128, "01", "psnr", "PSNR vs the fp32 .pte (dB)", 30.0, {}),
    "ssdlite320_mobilenetv3": ("general", 320, "01", "detect",
                               "fraction of top-20 detections agreeing", 0.90, {}),
    "yolox_s": ("general", 640, "255", "detect_yolox",
                "fraction of top-20 detections agreeing", 0.90, {"bgr": True}),
}


def detect_agreement_ssdlite(out32, out8, topk=20):
    """SSDLite emits 12 raw heads, (cls, box) per level. Compare which anchors the
    two builds would actually fire on: take the top-k class logits per level and
    measure the overlap of the chosen (anchor, class) pairs."""
    agree, total = 0, 0
    for i in range(0, len(out32), 2):
        a, b = out32[i].flatten(), out8[i].flatten()
        k = min(topk, a.numel())
        sa = set(torch.topk(a, k).indices.tolist())
        sb = set(torch.topk(b, k).indices.tolist())
        agree += len(sa & sb)
        total += k
    return agree / max(1, total)


def detect_agreement_yolox(out32, out8, topk=20):
    """YOLOX gives [1,8400,85]: objectness * best class picks the candidates."""
    def score(o):
        o = o[0]
        return o[:, 4] * o[:, 5:].max(dim=1).values
    sa = set(torch.topk(score(out32[0]), topk).indices.tolist())
    sb = set(torch.topk(score(out8[0]), topk).indices.tolist())
    return len(sa & sb) / topk


def audit(name):
    cat, size, norm, metric, unit, thr, kw = AUDITS[name]
    m32, m8 = _load(name, "fp32"), _load(name, "int8")
    batches = calib_loader(cat, size, norm, n=5, **kw)
    vals = []
    for b in batches:
        o32 = m32.execute(list(b))
        o8 = m8.execute(list(b))
        o32 = o32 if isinstance(o32, (list, tuple)) else [o32]
        o8 = o8 if isinstance(o8, (list, tuple)) else [o8]
        if metric == "psnr":
            vals.append(psnr(o32[0], o8[0]))
        elif metric == "iou":
            vals.append(mask_iou(o32[0], o8[0]))
        elif metric == "labels":
            vals.append(label_agreement(o32[0], o8[0]))
        elif metric == "detect":
            vals.append(detect_agreement_ssdlite(o32, o8))
        elif metric == "detect_yolox":
            vals.append(detect_agreement_yolox(o32, o8))
    vals.sort()
    med, worst = vals[len(vals) // 2], vals[0]
    verdict = "pass" if worst >= thr else "fail"
    print(f"{name}: {unit} median {med:.4f} worst {worst:.4f} -> {verdict}", flush=True)
    p = os.path.join(REPO, "results", f"{name}_int8.json")
    d = json.load(open(p))
    d["quality_override"] = {
        "metric": unit, "median": round(med, 4), "worst": round(worst, 4),
        "verdict": verdict,
        "why": (f"measured in the units that matter for this model: {unit}, "
                f"median {med:.4f} over five real images (worst {worst:.4f}) against "
                f"the fp32 build."),
    }
    json.dump(d, open(p, "w"), indent=2)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(AUDITS)):
        try:
            audit(n)
        except Exception as e:
            print(f"{n}: SKIP {type(e).__name__}: {str(e)[:100]}", flush=True)
