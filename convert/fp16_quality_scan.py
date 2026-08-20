import numpy as np, torch, glob, os
from PIL import Image
from executorch.runtime import Runtime

SPECS = {
    "u2net":                   (320,  "imagenet"),
    "dis_isnet":               (1024, "pm1"),
    "ormbg_isnet":             (1024, "zero_one"),
    "modnet_portrait_matting": (512,  "pm1"),
}
IMAGES = sorted(glob.glob("convert/calib_images/*/*.jpg"))[:6]

def prep(path, size, norm):
    x = np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR), np.float32)
    if norm == "imagenet":
        x = (x/255.0 - np.array([0.485,0.456,0.406], np.float32)) / np.array([0.229,0.224,0.225], np.float32)
    elif norm == "pm1":
        x = (x/255.0 - 0.5)/0.5
    else:
        x = x/255.0
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1)[None]))

def n01(a):
    lo, hi = a.min(), a.max()
    return (a-lo)/(hi-lo) if hi > lo else np.zeros_like(a)

rt = Runtime.get()
for name, (size, nrm) in SPECS.items():
    m32 = rt.load_program(f"pte/{name}_xnnpack_fp32.pte").load_method("forward")
    m16 = rt.load_program(f"pte/{name}_xnnpack_fp16.pte").load_method("forward")
    print(f"--- {name}")
    for p in IMAGES:
        inp = prep(p, size, nrm)
        a = m32.execute([inp])[0].numpy().reshape(-1)
        b = m16.execute([inp])[0].numpy().reshape(-1)
        na, nb = n01(a), n01(b)
        fa, fb = (na>0.5), (nb>0.5)
        iou = (fa&fb).sum()/max(1,(fa|fb).sum())
        print("  %-16s fp32 fg %5.1f%%  fp16 fg %5.1f%%  normalized IoU %.4f  corr %.4f"
              % (os.path.basename(p), fa.mean()*100, fb.mean()*100, iou,
                 np.corrcoef(a,b)[0,1] if a.std()>0 and b.std()>0 else float('nan')))
