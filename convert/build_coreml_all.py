"""Re-export the whole shelf through the Core ML backend.

XNNPACK is CPU-only. On Depth-Anything-V2 the Core ML backend measured 11.7x
faster on an iPhone 17 Pro at half the file size, so every model deserves the
same treatment — but not every graph will lower, and the point of this driver is
to find out which do without hand-running 25 scripts.

Each export script is re-run unchanged: `CONVERT_BACKEND=coreml` in the harness
swaps the partitioner and the output naming. Precision is always fp32 here
because Core ML picks its own compute precision (fp16) via the compile spec.

Usage: python convert/build_coreml_all.py [name-fragment ...]
Writes results/coreml_build_report.json.
"""
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERT = os.path.join(REPO, "convert")

# (script, extra args). The one-off DA2 Core ML script is skipped: the env var
# covers it now.
JOBS = [
    ("export_dinov2.py", []),
    ("export_depth_anything_v2.py", []),
    ("export_modnet.py", []),
    ("export_effnet_b1.py", []),
    ("export_clip.py", []),
    ("export_ssdlite.py", []),
    ("export_edsr.py", []),
    ("export_twinlite.py", []),
    ("export_pidnet.py", []),
    ("export_ormbg.py", []),
    ("export_sam21_tiny.py", []),
    ("export_rtdetrv2.py", []),
    ("export_dfine.py", []),
    ("export_yolox.py", []),
    ("export_edgetam.py", []),
    ("export_mobilesam.py", []),
    ("export_u2net.py", []),
    ("export_dis.py", []),
    ("export_real_esrgan.py", []),
    ("export_6drepnet.py", []),
    ("export_whisper_tiny.py", []),
    ("export_lama.py", []),
    ("export_moge2.py", []),
    ("export_rtmpose.py", ["--variant", "body"]),
    ("export_rtmpose.py", ["--variant", "hand"]),
    ("export_rtmpose.py", ["--variant", "face"]),
    ("export_rtmpose.py", ["--variant", "animal"]),
]


def main(filters):
    env = dict(os.environ, CONVERT_BACKEND="coreml")
    env.setdefault("CONVERT_REPOS", os.path.expanduser(
        "~/code/litertlm-convert/third_party"))
    report = []
    jobs = [j for j in JOBS
            if not filters or any(f in j[0] + " ".join(j[1]) for f in filters)]
    for i, (script, extra) in enumerate(jobs, 1):
        label = script[len("export_"):-3] + (f" {extra[-1]}" if extra else "")
        print(f"\n=== [{i}/{len(jobs)}] {label} ===", flush=True)
        t0 = time.time()
        p = subprocess.run([sys.executable, os.path.join(CONVERT, script), "fp32"] + extra,
                           capture_output=True, text=True, env=env, cwd=REPO)
        secs = round(time.time() - t0, 1)
        tail = (p.stdout + p.stderr).strip().splitlines()
        # The harness prints one summary line per converted model.
        summary = [l for l in tail if l.startswith("[") and ":coreml/" in l]
        deleg = [l for l in tail if l.strip().startswith("delegation:")]
        ok = p.returncode == 0 and bool(summary)
        for l in summary + deleg:
            print("  " + l.strip(), flush=True)
        if not ok:
            why = next((l for l in reversed(tail)
                        if l.strip() and not l.startswith(" ")), "(no output)")
            print(f"  FAILED after {secs}s: {why[:200]}", flush=True)
        report.append({"label": label, "script": script, "ok": ok,
                       "seconds": secs, "summary": summary, "delegation": deleg,
                       "error": None if ok else "\n".join(tail[-25:])})
        json.dump(report, open(os.path.join(REPO, "results",
                                            "coreml_build_report.json"), "w"), indent=2)
    n_ok = sum(r["ok"] for r in report)
    print(f"\n=== {n_ok}/{len(report)} converted ===")
    for r in report:
        if not r["ok"]:
            print(f"  FAILED {r['label']}")


if __name__ == "__main__":
    main(sys.argv[1:])
