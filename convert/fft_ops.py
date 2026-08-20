"""Export-safe replacements for the FFT ops that ExecuTorch cannot lower.

`torch.fft.rfftn` exports and runs fine. The inverse does not: building a complex
tensor hits `aten.complex`, which is not in the Core ATen opset, and routing around
it with `view_as_complex` lowers but fails at runtime. That blocks every model built
on Fast Fourier Convolution (LaMa and its descendants).

For a fixed spatial size the inverse transform is a fixed linear map, so it can be
written as two real matmuls against precomputed matrices — which is both exportable
and delegatable. At 512x512 the four matrices come to 3.1 MB.

    from fft_ops import IRFFT2
    irfft2 = IRFFT2(h, w)            # nn.Module holding the matrices as buffers
    y = irfft2(spec.real, spec.imag) # (..., H, W//2+1) -> (..., H, W)

Run this file directly to check the identity against torch.fft.irfftn.
"""
import math

import torch
import torch.nn as nn


class IRFFT2(nn.Module):
    """Inverse of `torch.fft.rfftn(x, dim=(-2,-1), norm=...)` for a fixed H x W.

    Takes the real and imaginary halves separately, so nothing in the graph ever
    holds a complex tensor. Two steps: a full inverse DFT along H (both parts
    needed, since the rows are still complex), then the real inverse rfft along W,
    where the Hermitian-symmetric bins count twice.
    """

    def __init__(self, h, w, norm="backward"):
        super().__init__()
        assert norm in ("backward", "ortho"), f"unsupported norm {norm!r}"
        wf = w // 2 + 1
        k = torch.arange(wf, dtype=torch.float64)
        n = torch.arange(w, dtype=torch.float64)
        ang = 2 * math.pi * torch.outer(n, k) / w
        weight = torch.full((wf,), 2.0, dtype=torch.float64)
        weight[0] = 1.0
        if w % 2 == 0:
            weight[-1] = 1.0
        m = torch.arange(h, dtype=torch.float64)
        ang_h = 2 * math.pi * torch.outer(m, m) / h
        # "ortho" splits the 1/N over the forward and inverse transforms.
        sh = 1.0 / math.sqrt(h) if norm == "ortho" else 1.0 / h
        sw = 1.0 / math.sqrt(w) if norm == "ortho" else 1.0 / w

        self.register_buffer("cos_h", (torch.cos(ang_h) * sh).float())
        self.register_buffer("sin_h", (torch.sin(ang_h) * sh).float())
        # transposed once here so forward() is a plain matmul
        self.register_buffer("cos_w", (torch.cos(ang) * weight * sw).float().t().contiguous())
        self.register_buffer("neg_sin_w", (-torch.sin(ang) * weight * sw).float().t().contiguous())

    def forward(self, real, imag):
        re_h = torch.matmul(self.cos_h, real) - torch.matmul(self.sin_h, imag)
        im_h = torch.matmul(self.sin_h, real) + torch.matmul(self.cos_h, imag)
        return torch.matmul(re_h, self.cos_w) + torch.matmul(im_h, self.neg_sin_w)


class RFFT2(nn.Module):
    """`torch.fft.rfftn(x, dim=(-2,-1), norm=...)` for a fixed H x W, as matmuls.

    `rfft2_parts` below is enough for XNNPACK: the complex tensor exists for one
    line before `.real`/`.imag` split it, and the delegate never sees it. Core ML
    is stricter — coremltools rejects a complex dtype outright — so the forward
    transform needs the same real-matrix treatment as the inverse.

    DFT is X[k] = sum_n x[n] (cos - i sin), so along W a real input gives
    re = x @ cos, im = -(x @ sin); then the H axis is a full complex DFT.
    """

    def __init__(self, h, w, norm="backward"):
        super().__init__()
        assert norm in ("backward", "ortho"), f"unsupported norm {norm!r}"
        wf = w // 2 + 1
        n = torch.arange(w, dtype=torch.float64)
        k = torch.arange(wf, dtype=torch.float64)
        ang_w = 2 * math.pi * torch.outer(n, k) / w
        m = torch.arange(h, dtype=torch.float64)
        ang_h = 2 * math.pi * torch.outer(m, m) / h
        # "backward" puts the whole 1/N on the inverse, so the forward is unscaled.
        sh = 1.0 / math.sqrt(h) if norm == "ortho" else 1.0
        sw = 1.0 / math.sqrt(w) if norm == "ortho" else 1.0

        self.register_buffer("cos_w", (torch.cos(ang_w) * sw).float())
        self.register_buffer("neg_sin_w", (-torch.sin(ang_w) * sw).float())
        self.register_buffer("cos_h", (torch.cos(ang_h) * sh).float())
        self.register_buffer("neg_sin_h", (-torch.sin(ang_h) * sh).float())

    def forward(self, x):
        re_w = torch.matmul(x, self.cos_w)
        im_w = torch.matmul(x, self.neg_sin_w)
        re = torch.matmul(self.cos_h, re_w) - torch.matmul(self.neg_sin_h, im_w)
        im = torch.matmul(self.neg_sin_h, re_w) + torch.matmul(self.cos_h, im_w)
        return re, im


def rfft2_parts(x, norm="backward"):
    """Forward half-spectrum as two real tensors, via torch's own FFT.

    Fine for XNNPACK; use `RFFT2` when the graph must never hold a complex tensor.
    """
    f = torch.fft.rfftn(x, dim=(-2, -1), norm=norm)
    return f.real, f.imag


if __name__ == "__main__":
    for norm in ("backward", "ortho"):
        for h, w in [(32, 32), (64, 48), (128, 128), (256, 256)]:
            x = torch.randn(2, 3, h, w)
            re, im = rfft2_parts(x, norm)
            got = IRFFT2(h, w, norm)(re, im)
            ref = torch.fft.irfftn(torch.complex(re, im), s=(h, w), dim=(-2, -1), norm=norm)
            fre, fim = RFFT2(h, w, norm)(x)
            fwd = max((fre - re).abs().max().item(), (fim - im).abs().max().item())
            rt = IRFFT2(h, w, norm)(fre, fim)
            print(f"{norm} {h}x{w}: inverse={(got - ref).abs().max().item():.3e} "
                  f"forward={fwd:.3e} roundtrip={(rt - x).abs().max().item():.3e}")
