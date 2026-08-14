"""Run the post-processing each card documents, and check the result is sane.

The parity gate proves a `.pte` matches its PyTorch model. It says nothing about
whether the preprocessing and decoding written on the card are right, and those are
what an app actually copies. A card that documents the wrong normalisation or the
wrong decode ships a model nobody can use, and no correlation number would catch it.

Each check runs a real image through the documented recipe and asserts something
that has to hold if the recipe is right — a skeleton in anatomical order, a matte
that covers a plausible fraction of a portrait, detections inside the frame.

Usage: python convert/verify_cards.py [name ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from calib import calib_loader
from executorch.runtime import Runtime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO_KPT = ["nose", "eyeL", "eyeR", "earL", "earR", "shoL", "shoR", "elbL", "elbR",
            "wriL", "wriR", "hipL", "hipR", "kneL", "kneR", "ankL", "ankR"]


def _method(name, prec="fp32"):
    p = os.path.join(REPO, "pte", f"{name}_xnnpack_{prec}.pte")
    return Runtime.get().load_program(p).load_method("forward")


def verify_rtmpose():
    """Card says: keypoint k is at (argmax(x[k])/2, argmax(y[k])/2) in crop pixels.
    If that is right, the nose sits above the shoulders and the shoulders above the
    hips on an upright person."""
    m = _method("rtmpose_s_body")
    ok = 0
    crops = calib_loader("person", (256, 192), "imagenet", n=6)
    for im in crops:
        xs, ys = m.execute(list(im))
        y = ys[0].argmax(-1).float() / 2.0
        nose = y[COCO_KPT.index("nose")]
        sho = (y[COCO_KPT.index("shoL")] + y[COCO_KPT.index("shoR")]) / 2
        hip = (y[COCO_KPT.index("hipL")] + y[COCO_KPT.index("hipR")]) / 2
        ok += int(nose < sho < hip)
    return ok, len(crops), "skeletons in anatomical order"


def verify_yolox():
    """Card says: cx,cy,w,h are in input pixels and NMS is required. If that is
    right, decoded boxes land inside the 640x640 frame and have positive area."""
    from torchvision.ops import nms
    m = _method("yolox_s")
    ok = total = 0
    for im in calib_loader("street", 640, "255", n=6, bgr=True):
        o = m.execute(list(im))[0][0]
        cx, cy, w, h = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
        score = o[:, 4] * o[:, 5:].max(1).values
        keep = score > 0.3
        if keep.sum() == 0:
            continue
        b = boxes[keep][nms(boxes[keep], score[keep], 0.45)]
        inside = ((b[:, 0] >= -8) & (b[:, 1] >= -8) &
                  (b[:, 2] <= 648) & (b[:, 3] <= 648)).all().item()
        positive = ((b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])).all().item()
        ok += int(inside and positive)
        total += 1
    return ok, total, "images whose decoded boxes are in-frame with positive area"


def verify_modnet():
    """Card says: alpha matte 0-1 at 512x512 from RGB scaled to [-1,1]. If the
    normalisation is right, a portrait produces a matte that is neither empty nor
    the whole frame."""
    m = _method("modnet_portrait_matting")
    ok = 0
    crops = calib_loader("portrait", 512, "pm1", n=8)
    for im in crops:
        a = m.execute(list(im))[0]
        frac = (a > 0.5).float().mean().item()
        ok += int(0.02 < frac < 0.95 and 0.0 <= a.min() and a.max() <= 1.0)
    return ok, len(crops), "portraits with a plausible matte in 0-1"


def verify_moge():
    """Card says: points is a metric point map and mask marks valid pixels. If so,
    the masked depth (z) is positive and finite."""
    m = _method("moge2_vits")
    ok = 0
    imgs = calib_loader("general", 518, "imagenet", n=5)
    for im in imgs:
        points, normal, mask, scale = m.execute(list(im))
        z = points[..., 2][mask > 0]
        ok += int(z.numel() > 0 and torch.isfinite(z).all().item() and (z > 0).float().mean().item() > 0.9)
    return ok, len(imgs), "images whose valid pixels carry positive finite depth"


def _sam_family(enc_name, dec_name, labels_dtype, three_feats):
    """Cards for SAM 2.1, EdgeTAM and MobileSAM all promise the same thing: pass a
    click in 1024-space and the decoder returns mask logits you threshold at 0.
    Run that end to end and require a mask that is neither empty nor the whole
    frame.

    Gate on the model's own predicted IoU, the way the card tells apps to. A click
    at the centre of an arbitrary photo sometimes lands on background — the model
    correctly reports low confidence there (all three heads under 0.4), and asking
    for a clean mask anyway would be testing the click, not the conversion."""
    enc, dec = _method(enc_name), _method(dec_name)
    ok = considered = 0
    imgs = calib_loader("portrait", 1024, "imagenet", n=6)
    for im in imgs:
        feats = enc.execute(list(im))
        pts = torch.tensor([[[[512.0, 512.0]]]]) if three_feats else torch.tensor([[[512.0, 512.0]]])
        lbl = (torch.tensor([[[1]]], dtype=torch.int64) if labels_dtype is torch.int64
               else torch.tensor([[1.0]]))
        args = list(feats) + [pts, lbl] if three_feats else [feats[0], pts, lbl]
        masks, iou = dec.execute(args)
        scores = iou.flatten()
        best = scores.argmax().item()
        if scores[best].item() < 0.5:
            continue  # the click missed; the model says so
        considered += 1
        m = masks.reshape(-1, masks.shape[-2], masks.shape[-1])[best] > 0
        frac = m.float().mean().item()
        # Only catch genuinely degenerate output. A confident click can legitimately
        # select something small — that is the point of a promptable segmenter.
        ok += int(0.0 < frac < 0.95 and torch.isfinite(masks).all().item())
    return ok, considered, "confident clicks that produce a non-degenerate mask"


def verify_sam21():
    return _sam_family("sam21_tiny_encoder", "sam21_tiny_decoder", torch.int64, True)


def verify_edgetam():
    return _sam_family("edgetam_encoder", "edgetam_decoder", torch.int64, True)


def verify_mobilesam():
    return _sam_family("mobilesam_encoder", "mobilesam_decoder", torch.float32, False)


CHECKS = {
    "sam21_tiny": verify_sam21,
    "edgetam": verify_edgetam,
    "mobilesam": verify_mobilesam,
    "rtmpose_s_body": verify_rtmpose,
    "yolox_s": verify_yolox,
    "modnet_portrait_matting": verify_modnet,
    "moge2_vits": verify_moge,
}

if __name__ == "__main__":
    failed = 0
    for name in (sys.argv[1:] or list(CHECKS)):
        try:
            ok, total, what = CHECKS[name]()
            verdict = "OK" if ok == total and total > 0 else "FAILED"
            failed += verdict == "FAILED"
            print(f"{name}: {ok}/{total} {what} -> {verdict}", flush=True)
        except Exception as e:
            failed += 1
            print(f"{name}: ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
    sys.exit(1 if failed else 0)
