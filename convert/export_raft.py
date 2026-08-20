"""RAFT-small (optical flow) -> ExecuTorch, XNNPACK and Core ML.

The shelf covers detection, segmentation, depth, geometry, pose, restoration and
ASR but has no motion model, and optical flow is the one every video pipeline
reaches for first. RAFT-small comes from torchvision, so the weights are BSD-3
like SSDLite's and there is no repository to clone or dependency to stub.

RAFT refines its estimate in a loop and torchvision returns every iterate. Only
the last one is the answer, so the wrapper returns it and the fixed iteration
count keeps the graph static.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from harness import convert_and_gate
from calib import calib_loader
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

# Divisible by 8: RAFT works on a 1/8-resolution correlation volume.
H, W = 384, 512
ITERS = 12


class RAFT(torch.nn.Module):
    """Two frames in, one flow field out."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, img1, img2):
        return self.m(img1, img2, num_flow_updates=ITERS)[-1]


net = raft_small(weights=Raft_Small_Weights.DEFAULT).eval()

# A real pair, not two unrelated photos: shifting one frame gives the small,
# coherent motion the model is built for, so the parity gate exercises the
# correlation volume rather than a field of noise.
frames = calib_loader("general", (H, W), "pm1", n=2)
img1 = frames[0][0]
img2 = torch.roll(img1, shifts=(3, 5), dims=(-2, -1)).contiguous()

for prec in (sys.argv[1:] or ["fp32", "fp16"]):
    convert_and_gate(
        "raft_small", RAFT(net), (torch.randn(1, 3, H, W), torch.randn(1, 3, H, W)),
        precision=prec,
        gate_inputs=(img1, img2),
        extra_meta={
            "source": "torchvision raft_small (C_T_V2, FlyingChairs + FlyingThings3D)",
            "license": "BSD-3-Clause",
            "preprocess": f"two RGB frames, each scaled to [-1,1], {H}x{W} "
                          f"(both dimensions must stay divisible by 8)",
            "outputs": f"flow [1,2,{H},{W}] in pixels: channel 0 is horizontal "
                       f"displacement from frame 1 to frame 2, channel 1 vertical. "
                       f"Refined over {ITERS} iterations, which are baked into the "
                       f"graph — the intermediate iterates are not returned.",
        },
    )
