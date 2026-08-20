"""LaMa (big-lama, resolution-robust inpainting) -> ExecuTorch XNNPACK .pte.

LaMa is built on Fast Fourier Convolution, and its `FourierUnit` ends with
`torch.complex(...)` followed by `torch.fft.irfftn` — the two lines ExecuTorch
cannot lower. Everything else about the model exports as-is. Patching those two
lines to use `fft_ops.IRFFT2`, which is the same transform written as real
matmuls, is the whole conversion.

The spatial size has to be fixed for that (the inverse matrices are per-size), so
this exports at 512x512. LaMa is fully convolutional and tolerates other sizes in
principle; re-run with --size to build another one.
"""
import sys, os, zipfile, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPOS = os.environ.get("CONVERT_REPOS", "")
sys.path.insert(0, os.path.join(REPOS, "lama"))
import torch
import torch.nn as nn
import yaml
from huggingface_hub import hf_hub_download
from harness import convert_and_gate
from calib import calib_loader
from fft_ops import IRFFT2, RFFT2

SIZE = 512
for i, a in enumerate(sys.argv):
    if a == "--size":
        SIZE = int(sys.argv[i + 1])

zip_path = hf_hub_download("smartywu/big-lama", "big-lama.zip")
work = os.path.join(tempfile.gettempdir(), "big-lama")
if not os.path.exists(os.path.join(work, "big-lama", "models", "best.ckpt")):
    zipfile.ZipFile(zip_path).extractall(work)
cfg = yaml.safe_load(open(os.path.join(work, "big-lama", "config.yaml")))

# `saicinpainting.utils` imports pytorch_lightning for `seed_everything` alone, and
# that import runs on the way to the generator. Installing Lightning to satisfy one
# unused function would drag a torch version pin into this venv; a stub is enough.
import importlib.abc
import importlib.machinery
import types


class _Stub(types.ModuleType):
    """Answers any attribute with a callable no-op, so `from x.y import Z` works
    for a package nothing here actually calls."""

    __path__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub(f"{self.__name__}.{name}")

    def __call__(self, *a, **k):
        return None


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """`import a.b.c` goes through the import machinery rather than getattr, so
    attribute stubbing alone is not enough — this answers the whole subtree.
    find_spec, not the find_module/load_module pair: those were removed in 3.12."""

    prefix = "pytorch_lightning"

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.prefix or fullname.startswith(self.prefix + "."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _Stub(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _StubFinder())

from saicinpainting.training.modules.ffc import FourierUnit, FFCResNetGenerator


def patch_fourier_unit(size):
    """Replace both FFTs in FourierUnit.forward with real-matrix equivalents.

    Each unit sees a different spatial size (LaMa downsamples three times), so the
    matrices are built lazily per size and cached on the module. Patching the
    class rather than instances keeps this correct through deepcopy — the harness
    copies the model for fp16/int8."""
    orig = FourierUnit.forward

    def forward(self, x):
        batch = x.shape[0]
        h, w = x.shape[-2:]
        # Forward transform as real matrices too, not just the inverse. XNNPACK
        # tolerates the complex tensor that rfftn produces because `.real`/`.imag`
        # split it on the next line, but coremltools rejects a complex dtype and
        # reports it as an unsupported model output, far from where it came from.
        fkey = f"_rfft_{h}x{w}"
        if not hasattr(self, fkey):
            setattr(self, fkey, RFFT2(h, w, norm=self.fft_norm).to(x.device))
        re, im = getattr(self, fkey)(x)
        ffted = torch.stack((re, im), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1) + ffted.size()[3:])
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        key = f"_irfft_{h}x{w}"
        if not hasattr(self, key):
            setattr(self, key, IRFFT2(h, w, norm=self.fft_norm).to(x.device))
        return getattr(self, key)(ffted[..., 0], ffted[..., 1])

    FourierUnit.forward = forward
    return orig


patch_fourier_unit(SIZE)
gkw = {k: v for k, v in cfg["generator"].items() if k != "kind"}
# the shipped config leaves OmegaConf interpolations unresolved
gkw["downsample_conv_kwargs"] = {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False}
gkw["resnet_conv_kwargs"] = {"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False}
gkw["init_conv_kwargs"] = {"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False}
gen = FFCResNetGenerator(**gkw)
state = torch.load(os.path.join(work, "big-lama", "models", "best.ckpt"),
                   map_location="cpu", weights_only=False)["state_dict"]
gen.load_state_dict({k.replace("generator.", ""): v for k, v in state.items()
                     if k.startswith("generator.")}, strict=True)
gen.eval()

# Build the lazily-created IRFFT2 modules before export, so they are real buffers
# in the exported graph rather than something created during tracing.
with torch.no_grad():
    gen(torch.zeros(1, 4, SIZE, SIZE))


class Inpaint(nn.Module):
    """image + mask -> inpainted image. LaMa takes them concatenated (4 channels)
    and composites the result itself, which keeps the app side to one call."""

    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, image, mask):
        x = torch.cat([image * (1 - mask), mask], dim=1)
        out = self.g(x)
        return (out * mask + image * (1 - mask)).contiguous()


mask = torch.zeros(1, 1, SIZE, SIZE)
mask[:, :, SIZE // 4:SIZE // 2, SIZE // 4:SIZE // 2] = 1.0
batches = [(b[0], mask) for b in calib_loader("general", SIZE, "01")]
cal = lambda mod: [mod(*b) for b in batches]

for prec in (sys.argv[1:] or ["fp32"]):
    if prec.startswith("--") or prec.isdigit():
        continue
    convert_and_gate(
        f"lama_{SIZE}", Inpaint(gen).eval(), (torch.rand(1, 3, SIZE, SIZE), mask), runs=3,
        precision=prec,
        calibrate=cal if prec == "int8" else None,
        gate_inputs=batches[0],
        extra_meta={
            "source": "advimman/lama + smartywu/big-lama weights",
            "license": "Apache-2.0",
            "preprocess": f"image RGB 0-1 [1,3,{SIZE},{SIZE}] + mask [1,1,{SIZE},{SIZE}] "
                          f"where 1 marks the region to fill",
            "outputs": f"inpainted image [1,3,{SIZE},{SIZE}] RGB 0-1, already composited "
                       f"with the untouched region",
            "notes": "The inverse FFT inside every FourierUnit is replaced with the real "
                     "matmul form from convert/fft_ops.py; ExecuTorch cannot lower "
                     "torch.fft.irfftn. Spatial size is fixed because those matrices are "
                     "built per size.",
        },
    )
