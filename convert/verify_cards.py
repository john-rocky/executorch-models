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


def _simcc_xy(m, im, n_kpt):
    xs, ys = m.execute(list(im))
    return (xs[0].argmax(-1).float() / 2.0, ys[0].argmax(-1).float() / 2.0,
            torch.minimum(xs[0].max(-1).values, ys[0].max(-1).values))


def verify_rtmpose_hand():
    """The hand model wants a crop around one hand, and this repository has no hand
    detector to make one — feeding it a person crop puts a whole body in frame, so
    an anatomical check would be testing the input, not the conversion. What can be
    checked without that data is the decode contract itself: 21 keypoints, both
    axes landing inside the crop, finite confidences. The card states the detector
    requirement; verifying the pose quality needs a hand dataset."""
    m = _method("rtmpose_m_hand")
    crops = calib_loader("person", (256, 256), "imagenet", n=8)
    ok = 0
    for im in crops:
        x, y, conf = _simcc_xy(m, im, 21)
        ok += int(x.numel() == 21 and y.numel() == 21 and
                  0 <= x.min() and x.max() < 256 and 0 <= y.min() and y.max() < 256 and
                  torch.isfinite(conf).all().item())
    return ok, len(crops), "crops decoding to 21 in-frame keypoints"


def verify_rtmpose_face():
    """106-point face: the contour runs 0-32 along the jaw, and the eyes are
    points 66-83. Eyes above the jawline is true of any upright face."""
    m = _method("rtmpose_m_face")
    crops = calib_loader("portrait", (256, 256), "imagenet", n=8)
    ok = considered = 0
    for im in crops:
        x, y, conf = _simcc_xy(m, im, 106)
        if conf.median().item() < 0.3:
            continue
        considered += 1
        ok += int(y[66:84].mean().item() < y[0:33].mean().item())
    return ok, considered, "faces with the eyes above the jaw contour"


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


def verify_clip():
    """Card says: L2-normalise both embeddings, then cosine-match, with the text
    tower taking CLIP BPE tokens padded to 77 plus an attention mask. The whole
    contract — including the tokenisation, which nothing else exercises — is only
    right if a true caption outranks a false one on a real photo. Portrait crops
    are people, so "a photo of a person" has to beat "a photo of a car"."""
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    img_m, txt_m = _method("clip_vit_b32_image"), _method("clip_vit_b32_text")
    prompts = ["a photo of a person", "a photo of a car"]
    embs = []
    for p in prompts:
        enc = tok(p, padding="max_length", max_length=77, return_tensors="pt")
        e = txt_m.execute([enc["input_ids"], enc["attention_mask"]])[0]
        embs.append(torch.nn.functional.normalize(e, dim=-1))
    text = torch.cat(embs)
    ok = 0
    imgs = calib_loader("portrait", 224, "clip", n=8)
    for im in imgs:
        v = torch.nn.functional.normalize(img_m.execute(list(im))[0], dim=-1)
        ok += int((v @ text.T).argmax().item() == 0)
    return ok, len(imgs), "portraits where 'a person' outranks 'a car'"


def verify_whisper():
    """Card says: seed decoder_input_ids with
    [<|startoftranscript|>, <|lang|>, <|transcribe|>, <|notimestamps|>], take the
    argmax of row len-1, append, re-run, and stop at <|endoftext|>.

    Silence is the one input whose correct transcript is knowable without a speech
    dataset, so that is what this feeds: the loop has to terminate at <|endoftext|>
    rather than run to the window limit, and every token it emits along the way has
    to be a real vocabulary id. It checks the mechanics of the documented loop, not
    transcription accuracy."""
    enc, dec = _method("whisper_tiny_encoder"), _method("whisper_tiny_decoder")
    START, EOT, VOCAB, MAXT = 50258, 50257, 51865, 128
    PREFIX = [START, 50259, 50359, 50363]  # start, <|en|>, <|transcribe|>, <|notimestamps|>
    hidden = enc.execute([torch.zeros(1, 80, 3000)])[0]
    ids = torch.full((1, MAXT), EOT, dtype=torch.long)
    for i, t in enumerate(PREFIX):
        ids[0, i] = t
    n = len(PREFIX)
    emitted = []
    for _ in range(24):
        logits = dec.execute([hidden, ids])[0]
        nxt = int(logits[0, n - 1].argmax().item())
        emitted.append(nxt)
        if nxt == EOT or n >= MAXT:
            break
        ids[0, n] = nxt
        n += 1
    terminated = emitted and emitted[-1] == EOT
    in_vocab = all(0 <= t < VOCAB for t in emitted)
    return int(bool(terminated and in_vocab)), 1, "greedy loop terminating at <|endoftext|>"


def _mask_model(name, cat, size, norm):
    """ormbg, DIS and U^2-Net all promise an alpha mask in 0-1 at a stated size
    under a stated normalisation. If the normalisation on the card were wrong the
    mask would come out empty or saturated, so require it to be neither and to
    actually lie inside 0-1."""
    m = _method(name)
    imgs = calib_loader(cat, size, norm, n=6)
    ok = considered = 0
    for im in imgs:
        a = m.execute(list(im))[0]
        in_range = 0.0 <= a.min().item() and a.max().item() <= 1.0
        # A photo with nothing salient in it correctly yields an empty mask, so
        # only judge the ones where the model found something. What must never
        # happen is a mask covering everything, or values outside 0-1 — the
        # signature of a wrong normalisation or a doubled sigmoid.
        if a.max().item() < 0.5:
            continue
        considered += 1
        frac = (a > 0.5).float().mean().item()
        ok += int(in_range and frac < 0.95)
    return ok, considered, "images with an in-range, non-saturated mask"


def verify_ormbg():
    return _mask_model("ormbg_isnet", "portrait", 1024, "01")


def verify_dis():
    return _mask_model("dis_isnet", "general", 1024, "pm1")


def verify_u2net():
    return _mask_model("u2net", "general", 320, "imagenet")


def verify_depth_anything():
    """Card says: relative inverse depth. Inverse means nearer is larger, so on
    ordinary photographs the bottom of the frame — usually the ground closest to
    the camera — should read larger than the top."""
    m = _method("depth_anything_v2_small")
    imgs = calib_loader("street", 518, "imagenet", n=6)
    votes = 0
    for im in imgs:
        d = m.execute(list(im))[0][0]
        h = d.shape[0]
        if not torch.isfinite(d).all().item():
            return 0, len(imgs), "photos with finite depth (a frame was not finite)"
        votes += int(d[int(h * 0.8):].mean().item() > d[:int(h * 0.2)].mean().item())
    # Majority, not unanimity: "the bottom of the frame is nearer" is true of most
    # street photographs but not all of them, and the check is about the sign
    # convention on the card, not about any single picture.
    return int(votes * 2 > len(imgs)), 1, "a majority reading nearer at the bottom (inverse depth)"


def verify_edsr():
    """Card says: 128x128 tile in, 512x512 out, RGB 0-1. Check the scale factor and
    that the result stays in range — an upscaler that clips or explodes is useless."""
    m = _method("edsr_base_x4")
    imgs = calib_loader("general", 128, "01", n=4)
    ok = 0
    for im in imgs:
        y = m.execute(list(im))[0]
        ok += int(tuple(y.shape[-2:]) == (512, 512) and
                  y.min().item() > -0.2 and y.max().item() < 1.2)
    return ok, len(imgs), "tiles upscaled 4x and staying near 0-1"


CHECKS = {
    "whisper_tiny": verify_whisper,
    "ormbg_isnet": verify_ormbg,
    "dis_isnet": verify_dis,
    "u2net": verify_u2net,
    "depth_anything_v2_small": verify_depth_anything,
    "edsr_base_x4": verify_edsr,
    "clip_vit_b32": verify_clip,
    "sam21_tiny": verify_sam21,
    "edgetam": verify_edgetam,
    "mobilesam": verify_mobilesam,
    "rtmpose_s_body": verify_rtmpose,
    "rtmpose_m_hand": verify_rtmpose_hand,
    "rtmpose_m_face": verify_rtmpose_face,
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
