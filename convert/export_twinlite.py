"""TwinLiteNet (drivable area + lane) -> ExecuTorch XNNPACK .pte. Stock ConvTranspose2d kept."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPOS = os.environ.get("CONVERT_REPOS", "")
sys.path.insert(0, os.path.join(REPOS, "TwinLiteNet"))
import torch
from harness import convert_and_gate
from model.TwinLite import TwinLiteNet as Net

# XNNPACK PReLU segfaults at execute on macOS arm64 (executorch 1.4.0, repro:
# lone nn.PReLU(1) -> XnnpackPartitioner -> SIGSEGV). Exclude PReLU from
# delegation; it falls back to the portable kernel.
from executorch.backends.xnnpack.partition.config import ALL_PARTITIONER_CONFIGS
from executorch.backends.xnnpack.partition.config.node_configs import PreluConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

no_prelu = XnnpackPartitioner(configs=[c for c in ALL_PARTITIONER_CONFIGS if c is not PreluConfig])

net = Net()
sd = torch.load(os.path.join(REPOS, "TwinLiteNet", "pretrained", "best.pth"), map_location="cpu")
sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
print("load:", net.load_state_dict(sd, strict=False))

convert_and_gate(
    "twinlitenet",
    net.eval(),
    (torch.randn(1, 3, 360, 640),),
    partitioner=no_prelu,
    extra_meta={
        "source": "chequanghuy/TwinLiteNet (pretrained/best.pth)",
        "license": "MIT",
        "preprocess": "RGB 0-1, 360x640",
        "outputs": "drivable area [1,2,360,640] + lane line [1,2,360,640]",
    },
)
