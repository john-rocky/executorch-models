"""Ship converted .pte models to HF (mlboydaisuke namespace) with cards as README.

Vision repos list result names, not files: every precision variant that passed
its quality gate (see convert/variants.py) ships, so adding fp16/int8 is a
re-run of the export script, not an edit here.
"""
import os
import sys

from huggingface_hub import HfApi

from variants import label, load_variants

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTE = os.path.join(REPO, "pte")

# (repo_name, license_id, [result_name...] or [(local, remote)...], card, tags, base_model)
SHIP = [
    ("EfficientNet-B1-ExecuTorch", "bsd-3-clause", ["efficientnet_b1"],
     "cards/efficientnet_b1.md", ["image-classification"], None),
    ("DINOv2-ViT-S14-ExecuTorch", "apache-2.0", ["dinov2_vits14"],
     "cards/dinov2_vits14.md", ["image-feature-extraction"], "facebook/dinov2-small"),
    ("Depth-Anything-V2-Small-ExecuTorch", "apache-2.0", ["depth_anything_v2_small"],
     "cards/depth_anything_v2_small.md", ["depth-estimation"], "depth-anything/Depth-Anything-V2-Small-hf"),
    ("CLIP-ViT-B32-ExecuTorch", "mit", ["clip_vit_b32_image", "clip_vit_b32_text"],
     "cards/clip_vit_b32_image.md", ["zero-shot-image-classification"], "openai/clip-vit-base-patch32"),
    ("SSDLite320-MobileNetV3-ExecuTorch", "bsd-3-clause", ["ssdlite320_mobilenetv3"],
     "cards/ssdlite320_mobilenetv3.md", ["object-detection"], None),
    ("EDSR-x4-ExecuTorch", "apache-2.0", ["edsr_base_x4"],
     "cards/edsr_base_x4.md", ["image-to-image", "super-resolution"], "eugenesiow/edsr-base"),
    ("MODNet-ExecuTorch", "apache-2.0", ["modnet_portrait_matting"],
     "cards/modnet_portrait_matting.md", ["image-segmentation", "portrait-matting"], None),
    ("TwinLiteNet-ExecuTorch", "mit", ["twinlitenet"],
     "cards/twinlitenet.md", ["image-segmentation", "lane-detection"], None),
    ("PIDNet-S-Cityscapes-ExecuTorch", "mit", ["pidnet_s_cityscapes"],
     "cards/pidnet_s_cityscapes.md", ["image-segmentation"], None),
    ("ormbg-ExecuTorch", "apache-2.0", ["ormbg_isnet"],
     "cards/ormbg_isnet.md", ["image-segmentation", "background-removal"], "schirrmacher/ormbg"),
    ("Qwen3.5-0.8B-ExecuTorch", "apache-2.0",
     [(f"{REPO}/qwen3_5_0_8b_xnnpack_8da4w_e8.pte", "qwen3_5_0_8b_xnnpack_8da4w_e8.pte")],
     "cards/qwen3_5_0_8b_8da4w_e8.md", ["text-generation"], "Qwen/Qwen3.5-0.8B"),
    ("LFM2.5-350M-ExecuTorch", "other",
     [(f"{REPO}/lfm2_5_350m_xnnpack_8da4w.pte", "lfm2_5_350m_xnnpack_8da4w.pte")],
     "cards/lfm2_5_350m_8da4w.md", ["text-generation"], "LiquidAI/LFM2.5-350M"),
    ("LFM2.5-1.2B-Instruct-ExecuTorch", "other",
     [(f"{REPO}/lfm2_5_1_2b_xnnpack_8da4w.pte", "lfm2_5_1_2b_xnnpack_8da4w.pte")],
     "cards/lfm2_5_1_2b_8da4w.md", ["text-generation"], "LiquidAI/LFM2.5-1.2B-Instruct"),
    ("SAM2.1-hiera-tiny-ExecuTorch", "apache-2.0",
     ["sam21_tiny_encoder", "sam21_tiny_decoder"],
     "cards/sam21_tiny.md", ["mask-generation"], "facebook/sam2.1-hiera-tiny"),
    ("RT-DETRv2-S-ExecuTorch", "apache-2.0", ["rtdetrv2_s_r18vd"],
     "cards/rtdetrv2_s_r18vd.md", ["object-detection"], "PekingU/rtdetr_v2_r18vd"),
    ("D-FINE-S-ExecuTorch", "apache-2.0", ["dfine_s_coco"],
     "cards/dfine_s_coco.md", ["object-detection"], "ustc-community/dfine-small-coco"),
    ("YOLOX-s-ExecuTorch", "apache-2.0", ["yolox_s"],
     "cards/yolox_s.md", ["object-detection"], None),
    ("EdgeTAM-ExecuTorch", "apache-2.0", ["edgetam_encoder", "edgetam_decoder"],
     "cards/edgetam.md", ["mask-generation"], "facebook/EdgeTAM"),
    ("MobileSAM-ExecuTorch", "apache-2.0", ["mobilesam_encoder", "mobilesam_decoder"],
     "cards/mobilesam.md", ["mask-generation"], None),
]


def resolve_files(entries):
    """result names -> every gate-passing variant's .pte; (local, remote) passes through."""
    files = []
    for e in entries:
        if isinstance(e, tuple):
            files.append(e)
            continue
        variants = load_variants(e)
        assert variants, f"no results for {e!r} — run its export script first"
        for prec, v in variants.items():
            if not v["ships"]:
                if v["skip_reason"] == "quality":
                    why = f"corr {v['worst_corr']:.3f} below gate"
                elif v["skip_reason"] == "dominated":
                    why = f"no smaller than {v['dominated_by']}"
                else:
                    why = f"{100 * v['size_ratio']:.0f}% of fp32 size, no gain"
                print(f"  skip {e} {label(prec, v)}: {why}")
                continue
            files.append((os.path.join(PTE, v["pte"]), v["pte"]))
    return files

CLIP_EXTRA = """
This repo holds **both towers**: `clip_vit_b32_image_xnnpack_fp32.pte` (image) and
`clip_vit_b32_text_xnnpack_fp32.pte` (text, fixed len 77 + attention mask).
L2-normalize both embeddings, then cosine-match.
"""


def main(only=None, dry_run=False):
    api = HfApi()
    user = api.whoami()["name"]
    for name, lic, entries, card, tags, base in SHIP:
        if only and name != only:
            continue
        repo_id = f"{user}/{name}"
        files = resolve_files(entries)
        for local, _ in files:
            assert os.path.exists(local), f"missing {local}"
        if dry_run:
            total = sum(os.path.getsize(l) for l, _ in files) / 1e6
            print(f"{repo_id}: {len(files)} file(s), {total:.1f}MB total")
            for _, remote in files:
                print(f"    {remote}")
            continue
        fm = ["---", f"license: {lic}",
              "tags:", "- executorch", "- xnnpack", "- pte", "- on-device"]
        fm += [f"- {t}" for t in tags]
        if base:
            fm += ["base_model:", f"- {base}"]
        fm += ["---", ""]
        body = open(os.path.join(REPO, card)).read()
        if name.startswith("CLIP"):
            body += CLIP_EXTRA
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=("\n".join(fm) + body).encode(),
                        path_in_repo="README.md", repo_id=repo_id)
        for local, remote in files:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo_id)
        print(f"shipped {repo_id} ({len(files)} file(s))")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    main(args[0] if args else None, dry_run="--dry-run" in sys.argv)
