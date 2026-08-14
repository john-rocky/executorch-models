"""Load calibration images as [1,3,H,W] fp32 batches for PT2E int8 calibration.

Usage in an export script:
    from calib import calib_loader
    cal = lambda m: [m(*b) for b in calib_loader("street", 1024, "imagenet")]
    convert_and_gate(..., precision="int8", calibrate=cal)
"""
import glob
import os

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

NORMS = {
    "imagenet": (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
    "clip": (np.array([0.48145466, 0.4578275, 0.40821073]),
             np.array([0.26862954, 0.26130258, 0.27577711])),
    "pm1": (np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])),  # [-1, 1]
    "01": (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
    "255": (np.array([0.0, 0.0, 0.0]), np.array([1 / 255.0] * 3)),  # raw 0..255
}


def calib_loader(category, size, norm="imagenet", n=10, bgr=False):
    """size: int (square) or (h, w). bgr: channel-flip for BGR-input models (YOLOX)."""
    mean, std = NORMS[norm]
    h, w = (size, size) if isinstance(size, int) else size
    paths = sorted(glob.glob(os.path.join(HERE, "calib_images", category, "*")))[:n]
    assert paths, f"no calib images for {category!r} — run convert/fetch_calib.py"
    batches = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((w, h), Image.BILINEAR)
        a = (np.asarray(img).astype(np.float32) / 255.0 - mean) / std
        if bgr:
            a = a[:, :, ::-1]
        # ET runtime silently misreads non-contiguous tensors — keep the copy.
        t = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None].float()
        batches.append((t,))
    return batches
