"""Core ML delegate against XNNPACK, on the same phone, from the recorded device runs.

Reads results/device_ios_coreml_sweep.json and results/device_ios_coreml_vs_xnnpack.json rather
than any figure typed in by hand, so the chart cannot drift from what was measured.

    python convert/chart_delegate.py            # writes findings/coreml_vs_xnnpack.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# label prefix in the JSON -> how the model is named on the shelf
NAMES = {
    "da2": "Depth Anything V2-S",
    "dinov2": "DINOv2 ViT-S/14",
    "ormbg": "ORMBG",
    "u2net": "U²-Net",
    "modnet": "MODNet",
    "edgetam": "EdgeTAM encoder",
    "yolox": "YOLOX-s",
}

INK = "#12161c"
PAPER = "#ffffff"
CPU = "#9aa4b2"
ANE = "#1f9d6b"


def rows():
    """(name, xnnpack ms, core ml ms, xnnpack MB, core ml MB) for every model measured on both."""
    seen = {}
    for name in ("device_ios_coreml_sweep", "device_ios_coreml_vs_xnnpack"):
        path = os.path.join(REPO, "results", f"{name}.json")
        for r in json.load(open(path)):
            label = r["label"]
            key = label.split("-")[0]
            # da2 carries three builds; the Neural-Engine-only one duplicates the general build.
            if label.endswith("-ne"):
                continue
            kind = "cml" if ("coreml" in label or label.endswith("-cml")) else "xnn"
            seen.setdefault(key, {})[kind] = (r["median_ms"], r["size_mb"])

    out = []
    for key, pair in seen.items():
        if "xnn" not in pair or "cml" not in pair:
            continue
        out.append((NAMES.get(key, key), pair["xnn"][0], pair["cml"][0],
                    pair["xnn"][1], pair["cml"][1]))
    # Slowest on the CPU first, so the eye starts where the delegate matters most.
    return sorted(out, key=lambda r: -r[1])


def main():
    data = rows()
    ratios = sorted(r[1] / r[2] for r in data)
    median = ratios[len(ratios) // 2]

    fig, ax = plt.subplots(figsize=(11, 0.78 * len(data) + 2.4), facecolor=PAPER)
    ax.set_facecolor(PAPER)

    height = 0.34
    # Each row is drawn to its own CPU time, so within a row the two bars are exactly to scale and
    # the green one's length *is* the ratio. A log axis would have fitted every model against one
    # ruler, but on a log axis a bar eleven times shorter does not look eleven times shorter, which
    # is the one thing a bar is for. Across rows, read the milliseconds instead of the lengths.
    outside = ax.get_yaxis_transform()
    for i, (name, xnn, cml, xnn_mb, cml_mb) in enumerate(data):
        y = len(data) - 1 - i
        share = cml / xnn
        ax.barh(y + height / 1.7, 1.0, height, color=CPU, zorder=3)
        ax.barh(y - height / 1.7, share, height, color=ANE, zorder=3)
        ax.text(1.0 - 0.008, y + height / 1.7, f"{xnn:,.0f} ms", va="center", ha="right",
                fontsize=10.5, color="white", fontweight="bold", zorder=4)
        ax.text(share + 0.012, y - height / 1.7, f"{cml:,.1f} ms", va="center", ha="left",
                fontsize=10.5, color=ANE, fontweight="bold", zorder=4)
        ax.text(1.035, y + 0.13, f"{xnn / cml:.1f}×", transform=outside, va="center", ha="left",
                fontsize=15, color=INK, fontweight="bold", clip_on=False)
        ax.text(1.035, y - 0.26, f"{xnn_mb:,.0f} → {cml_mb:,.0f} MB", transform=outside,
                va="center", ha="left", fontsize=9, color="#9ca3af", clip_on=False)

    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([r[0] for r in reversed(data)], fontsize=12.5, color=INK)
    ax.set_ylim(-0.7, len(data) - 0.3)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "¼", "½", "¾", "the CPU's time"], fontsize=10.5, color="#9ca3af")
    ax.grid(axis="x", color="#eef0f3", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)

    fig.text(0.012, 0.965, "The same .pte, the same iPhone, one line of export changed",
             fontsize=17.5, color=INK, fontweight="bold", va="top")
    fig.text(0.012, 0.905,
             "ExecuTorch on iOS: Core ML delegate against the XNNPACK CPU backend. "
             f"Median of 10 runs each, iPhone 17 Pro, iOS 27.\nEvery row is drawn to its own CPU "
             "time, so the green bar's length is the speedup. Median "
             f"{median:.1f}×, at roughly half the file size.",
             fontsize=11.5, color="#4b5563", va="top", linespacing=1.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=CPU), plt.Rectangle((0, 0), 1, 1, color=ANE)]
    fig.legend(handles, ["XNNPACK · CPU", "Core ML · Neural Engine"],
               loc="lower left", bbox_to_anchor=(0.012, 0.005), frameon=False,
               fontsize=11.5, ncol=2, handlelength=1.1, handleheight=1.1)

    fig.subplots_adjust(left=0.185, right=0.845, top=0.80, bottom=0.115)
    dest = os.path.join(REPO, "findings", "coreml_vs_xnnpack.png")
    fig.savefig(dest, dpi=170, facecolor=PAPER)
    print(dest)
    for name, xnn, cml, xnn_mb, cml_mb in data:
        print(f"  {name:24s} {xnn:8.1f} -> {cml:6.1f} ms  {xnn / cml:5.1f}x")
    print(f"  median {median:.1f}x over {len(data)} models")


if __name__ == "__main__":
    main()
