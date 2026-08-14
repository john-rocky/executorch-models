"""Build a person-crop calibration set for the top-down pose models.

RTMPose consumes a crop around one person, not a scene. Calibrating or auditing it
on whole photographs measures nothing: the fp32 model itself produces no confident
keypoints on any image in the street or general sets. Rather than hunt for a
crop dataset, run the detector already on the shelf over those photos and keep what
it finds — the shelf validating itself.

Writes convert/calib_images/person/ (gitignored; rerun to regenerate).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from PIL import Image
from executorch.runtime import Runtime
from torchvision.ops import nms

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "calib_images", "person")
SRC = [os.path.join(HERE, "calib_images", c) for c in ("street", "portrait")]
SIZE = 640
PERSON = 0  # COCO class id in YOLOX's ordering


def detect_people(method, img):
    """YOLOX at 640, BGR 0..255, no normalisation."""
    a = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)).astype(np.float32)
    x = torch.from_numpy(np.ascontiguousarray(a[:, :, ::-1].transpose(2, 0, 1)))[None]
    o = method.execute([x])[0][0]
    cx, cy, w, h = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
    cls = o[:, 5:]
    score = o[:, 4] * cls.max(1).values
    keep = (score > 0.35) & (cls.argmax(1) == PERSON)
    boxes, score = boxes[keep], score[keep]
    if boxes.numel() == 0:
        return []
    return boxes[nms(boxes, score, 0.45)].tolist()


def main(limit=12):
    os.makedirs(OUT, exist_ok=True)
    pte = os.path.join(os.path.dirname(HERE), "pte", "yolox_s_xnnpack_fp32.pte")
    method = Runtime.get().load_program(pte).load_method("forward")
    n = 0
    for d in SRC:
        for f in sorted(os.listdir(d)):
            if n >= limit:
                break
            img = Image.open(os.path.join(d, f)).convert("RGB")
            sx, sy = img.width / SIZE, img.height / SIZE
            for (x1, y1, x2, y2) in detect_people(method, img):
                if n >= limit:
                    break
                # widen a little: a detector box is tighter than the crop a pose
                # model expects, and RTMPose is sensitive to that framing
                bx1, by1 = x1 * sx, y1 * sy
                bx2, by2 = x2 * sx, y2 * sy
                mw, mh = (bx2 - bx1) * 0.15, (by2 - by1) * 0.1
                box = (max(0, bx1 - mw), max(0, by1 - mh),
                       min(img.width, bx2 + mw), min(img.height, by2 + mh))
                if box[2] - box[0] < 48 or box[3] - box[1] < 96:
                    continue  # too small to carry pose signal
                img.crop(box).save(os.path.join(OUT, f"person_{n:02d}.jpg"), quality=95)
                n += 1
    print(f"wrote {n} person crops -> {OUT}")


if __name__ == "__main__":
    main()
