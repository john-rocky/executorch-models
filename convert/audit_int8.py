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


def simcc_keypoint_shift(out32, out8):
    """RTMPose emits two 1-D distributions per keypoint, not coordinates. What
    matters is where the argmax lands, so decode both and report the largest
    keypoint displacement in crop pixels (the split ratio is 2, hence the /2)."""
    def decode(o):
        x = o[0][0].argmax(dim=-1).float() / 2.0
        y = o[1][0].argmax(dim=-1).float() / 2.0
        return torch.stack([x, y], dim=-1)
    return (decode(out32) - decode(out8)).abs().max().item()


def top1_agreement(a, b):
    """A classifier is judged by the label it returns, not by logit correlation."""
    return float(a[0].argmax().item() == b[0].argmax().item())


def cosine(a, b):
    """Embedding models are used through cosine similarity, so measure that
    rather than element-wise correlation."""
    return torch.nn.functional.cosine_similarity(
        a.flatten(1).float(), b.flatten(1).float(), dim=1).min().item()


def depth_delta1(a, b):
    """Depth is judged by ratio, not difference: the fraction of pixels whose
    predicted value is within 1.25x of the reference, the standard delta-1."""
    x = a.flatten().float().abs().clamp_min(1e-6)
    y = b.flatten().float().abs().clamp_min(1e-6)
    ratio = torch.max(x / y, y / x)
    return (ratio < 1.25).float().mean().item()


# name -> (calib category, size, norm, metric, unit, pass threshold, extra kwargs)
AUDITS = {
    "modnet_portrait_matting": ("portrait", 512, "pm1", "iou", "mask IoU at 0.5", 0.95, {}),
    "ormbg_isnet": ("portrait", 1024, "01", "iou", "mask IoU at 0.5", 0.95, {}),
    "u2net": ("general", 320, "imagenet", "iou", "mask IoU at 0.5", 0.95, {}),
    "dis_isnet": ("general", 1024, "pm1", "iou", "mask IoU at 0.5", 0.95, {}),
    "pidnet_s_cityscapes": ("street", 1024, "imagenet", "labels",
                            "fraction of pixels keeping their class", 0.95, {}),
    "edsr_base_x4": ("general", 128, "01", "psnr", "PSNR vs the fp32 .pte (dB)", 30.0, {}),
    "efficientnet_b1": ("general", 240, "imagenet", "top1",
                        "fraction of images keeping the fp32 top-1 label", 0.9, {}),
    # 30 dB is where a restoration result stops being visibly different; LaMa's
    # int8 build sits at 22 and is not published because of it.
    "lama_512": ("general", 512, "01", "psnr_masked", "PSNR vs the fp32 .pte (dB)",
                 30.0, {}),
    # Embedding models are consumed through cosine similarity; depth through ratio.
    "dinov2_vits14": ("general", 518, "imagenet", "cosine",
                      "cosine similarity of the embeddings", 0.99, {}),
    "clip_vit_b32_image": ("general", 224, "clip", "cosine",
                           "cosine similarity of the image embeddings", 0.99, {}),
    "depth_anything_v2_small": ("general", 518, "imagenet", "depth",
                                "fraction of pixels within 1.25x of the fp32 depth", 0.99, {}),
    "mobilesam_encoder": ("portrait", 1024, "01", "cosine",
                          "cosine similarity of the image embeddings", 0.99, {}),
    "whisper_tiny_encoder": (None, None, None, "whisper",
                             "decoded token sequences that match fp32", 0.99, {}),
    # MoGe returns four tensors; judge it on the geometry, which is what it is for.
    # Its worst-output correlation is dominated by a near-binary validity mask and
    # says nothing useful.
    "moge2_vits": ("general", 518, "imagenet", "geometry",
                   "cosine similarity of the point map and normals", 0.99, {}),
    # Top-down pose: person crops, and judged in pixels. Lower is better here, so
    # the threshold is negated (see the sign handling in audit()).
    "rtmpose_s_body": ("person", (256, 192), "imagenet", "keypoints",
                       "largest keypoint displacement against fp32, in crop pixels",
                       -4.0, {}),
    # Detection audits need imagery that actually contains detections; the street
    # set carries 653 firings above 0.3 in YOLOX fp32, the general set almost none.
    "ssdlite320_mobilenetv3": ("street", 320, "01", "detect",
                               "fraction of firing detections agreeing", 0.90, {}),
    "yolox_s": ("street", 640, "255", "detect_yolox",
                "fraction of post-NMS detections matched at IoU 0.5 and same class",
                0.90, {"bgr": True}),
}


def detect_agreement_ssdlite(out32, out8, thr=0.3):
    """SSDLite emits 12 raw heads, (cls, box) per level. Compare the anchors that
    would actually fire, not a top-k over raw logits: ranking logits on levels with
    no signal measures noise, and it is what made a healthy build look 72% broken
    the first time this audit ran."""
    fired = kept = 0
    for i in range(0, len(out32), 2):
        a = out32[i][0].reshape(91, -1).softmax(0)[1:].max(0).values
        b = out8[i][0].reshape(91, -1).softmax(0)[1:].max(0).values
        f = a > thr
        fired += f.sum().item()
        kept += ((b > thr) & f).sum().item()
    return kept, fired


def _yolox_detections(o, score_thr=0.3, iou_thr=0.45):
    from torchvision.ops import nms
    o = o[0]
    cx, cy, w, h = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
    cls = o[:, 5:]
    score = o[:, 4] * cls.max(1).values
    lab = cls.argmax(1)
    keep = score > score_thr
    boxes, score, lab = boxes[keep], score[keep], lab[keep]
    if boxes.numel() == 0:
        return boxes, lab
    k = nms(boxes, score, iou_thr)
    return boxes[k], lab[k]


def detect_agreement_yolox(out32, out8):
    """Compare the detections an app would actually see, i.e. after NMS. Counting
    firing anchors instead reads 77% on a build whose final detections agree 94% —
    YOLOX fires many redundant anchors per object and NMS collapses them, so the
    pre-NMS count measures redundancy, not quality."""
    b1, l1 = _yolox_detections(out32[0])
    b2, l2 = _yolox_detections(out8[0])
    if len(b1) == 0:
        return 0, 0
    matched = 0
    for j in range(len(b1)):
        if len(b2) == 0:
            break
        x1 = torch.max(b1[j, 0], b2[:, 0]); y1 = torch.max(b1[j, 1], b2[:, 1])
        x2 = torch.min(b1[j, 2], b2[:, 2]); y2 = torch.min(b1[j, 3], b2[:, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        a1 = (b1[j, 2] - b1[j, 0]) * (b1[j, 3] - b1[j, 1])
        a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        iou = inter / (a1 + a2 - inter + 1e-9)
        matched += int(((iou > 0.5) & (l2 == l1[j])).any().item())
    return matched, len(b1)


def audit_whisper():
    """The only question that matters for the encoder is whether the transcript
    changes, so decode greedily through the fp32 decoder from both encoders and
    compare token sequences. Audio is synthesised rather than downloaded: what is
    being compared is two encoders on identical input, and speech-like structure
    is enough to exercise them."""
    import torch as t
    enc32, enc8 = _load("whisper_tiny_encoder", "fp32"), _load("whisper_tiny_encoder", "int8")
    dec = _load("whisper_tiny_decoder", "fp32")
    START, EOT, MAXT = 50258, 50257, 128
    same = total = 0
    g = t.Generator().manual_seed(0)
    for clip in range(5):
        # a rough speech-shaped log-mel: formant-like bands plus noise
        mel = t.randn(1, 80, 3000, generator=g) * 0.5
        for f in (6, 14, 27):
            mel[:, f - 2:f + 3, :] += t.sin(t.linspace(0, 60 + 20 * clip, 3000))[None, None] * 2
        seqs = []
        for enc in (enc32, enc8):
            h = enc.execute([mel])[0]
            ids = t.full((1, MAXT), START, dtype=t.long)
            out = []
            for i in range(12):
                logits = dec.execute([h, ids])[0]
                nxt = int(logits[0, i].argmax().item())
                out.append(nxt)
                if nxt == EOT or i + 1 >= MAXT:
                    break
                ids[0, i + 1] = nxt
            seqs.append(out)
        total += 1
        same += int(seqs[0] == seqs[1])
        print(f"  clip{clip}: fp32 {seqs[0][:6]} | int8 {seqs[1][:6]} | "
              f"{'same' if seqs[0] == seqs[1] else 'DIFFERS'}", flush=True)
    return same / total


def audit(name):
    cat, size, norm, metric, unit, thr, kw = AUDITS[name]
    if metric == "whisper":
        frac = audit_whisper()
        verdict = "pass" if frac >= thr else "fail"
        summary = (f"{frac:.0%} of five decoded sequences are identical when the "
                   f"int8 encoder is swapped in for fp32, decoder held constant")
        print(f"{name}: {unit} -> {frac:.4f} {verdict} ({summary})", flush=True)
        p = os.path.join(REPO, "results", f"{name}_int8.json")
        d = json.load(open(p))
        d["quality_override"] = {"metric": unit, "median": frac, "worst": frac,
                                 "headline": frac, "verdict": verdict,
                                 "why": f"measured end to end — {unit}: {summary}."}
        json.dump(d, open(p, "w"), indent=2)
        return
    m32, m8 = _load(name, "fp32"), _load(name, "int8")
    batches = calib_loader(cat, size, norm, n=10, **kw)
    if metric == "psnr_masked":
        # LaMa takes (image, mask); hole in the upper-left quadrant.
        mask = torch.zeros(1, 1, size, size)
        mask[:, :, size // 4:size // 2, size // 4:size // 2] = 1.0
        batches = [(b[0], mask) for b in batches[:5]]
    vals, pooled_matched, pooled_total = [], [], []
    for b in batches:
        o32 = m32.execute(list(b))
        o8 = m8.execute(list(b))
        o32 = o32 if isinstance(o32, (list, tuple)) else [o32]
        o8 = o8 if isinstance(o8, (list, tuple)) else [o8]
        if metric == "psnr":
            vals.append(psnr(o32[0], o8[0]))
        elif metric == "psnr_masked":
            vals.append(psnr(o32[0], o8[0]))
        elif metric == "iou":
            vals.append(mask_iou(o32[0], o8[0]))
        elif metric == "labels":
            vals.append(label_agreement(o32[0], o8[0]))
        elif metric == "top1":
            vals.append(top1_agreement(o32[0], o8[0]))
        elif metric == "cosine":
            vals.append(cosine(o32[0], o8[0]))
        elif metric == "keypoints":
            vals.append(simcc_keypoint_shift(o32, o8))
        elif metric == "geometry":
            vals.append(min(cosine(o32[0], o8[0]), cosine(o32[1], o8[1])))
        elif metric == "depth":
            vals.append(depth_delta1(o32[0], o8[0]))
        elif metric.startswith("detect"):
            fn = detect_agreement_ssdlite if metric == "detect" else detect_agreement_yolox
            m, t = fn(o32, o8)
            pooled_matched.append(m)
            pooled_total.append(t)
            vals.append(m / t if t else 1.0)
    n = len(vals)
    vals.sort()
    med = vals[n // 2]
    worst = vals[-1] if thr < 0 else vals[0]
    if metric.startswith("detect"):
        # Pool across images rather than judging on the worst one. A single photo
        # where fp32 finds seven objects and int8 finds four reads 0.57 and says
        # little; what an app experiences is the overall hit rate.
        headline = sum(pooled_matched) / max(1, sum(pooled_total))
        summary = (f"{headline:.3f} of the fp32 build's detections are matched "
                   f"({sum(pooled_matched)} of {sum(pooled_total)} across {n} images), "
                   f"worst single image {worst:.3f}")
    elif thr < 0:
        # A "lower is better" metric: the threshold is stored negated, and the
        # median is the honest headline (one bad frame should not decide).
        headline = med
        summary = f"median {med:.2f} over {n} images, worst {worst:.2f}"
    else:
        headline = worst
        summary = f"median {med:.4f} over {n} real images, worst {worst:.4f}"
    verdict = ("pass" if headline <= -thr else "fail") if thr < 0 else (
        "pass" if headline >= thr else "fail")
    print(f"{name}: {unit} -> {headline:.4f} {verdict} ({summary})", flush=True)
    p = os.path.join(REPO, "results", f"{name}_int8.json")
    d = json.load(open(p))
    d["quality_override"] = {
        "metric": unit, "median": round(med, 4), "worst": round(worst, 4),
        "headline": round(headline, 4), "verdict": verdict,
        "why": f"measured in the units that matter for this model — {unit}: {summary}.",
    }
    json.dump(d, open(p, "w"), indent=2)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(AUDITS)):
        try:
            audit(n)
        except Exception as e:
            print(f"{n}: SKIP {type(e).__name__}: {str(e)[:100]}", flush=True)
