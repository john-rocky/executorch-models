"""What does the ExecuTorch wrapper cost on top of Core ML?

The Core ML backend measured 11.7x faster than XNNPACK on device, which means
ExecuTorch's fast path on iOS *is* Core ML. That raises the question of whether
the ExecuTorch layer is worth keeping at all on Apple hardware, and the number
that decides it is the overhead of ET+CoreML against a plain .mlpackage running
the same graph.

Same torch.export program, two consumers:
  - executorch .pte with the Core ML partitioner, run through the ET runtime
  - the same ExportedProgram converted by coremltools, run through Core ML directly

Usage: python convert/coreml_overhead.py
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import coremltools as ct

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZE = 518
RUNS = 20


def median_ms(fn, runs=RUNS):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), min(ts), max(ts)


from transformers import AutoModelForDepthEstimation


class DepthWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).predicted_depth


model = DepthWrapper(AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf")).eval()
x = torch.randn(1, 3, SIZE, SIZE)
ep = torch.export.export(model, (x,)).run_decompositions({})

# --- native Core ML, same compile settings the ExecuTorch backend uses ---
mlmodel = ct.convert(
    ep,
    inputs=[ct.TensorType(name="x", shape=x.shape)],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS17,
)
path = "/tmp/da2_native.mlpackage"
mlmodel.save(path)
native = ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)
feed = {"x": x.numpy()}
n_med, n_min, n_max = median_ms(lambda: native.predict(feed))
native_out = list(native.predict(feed).values())[0].squeeze()

# --- the shipped ExecuTorch Core ML build ---
from executorch.runtime import Runtime
pte = os.path.join(REPO, "pte", "da2_coreml_all.pte")
method = Runtime.get().load_program(pte).load_method("forward")
e_med, e_min, e_max = median_ms(lambda: method.execute([x]))
et_out = np.asarray(method.execute([x])[0]).squeeze()

size_native = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fs in os.walk(path) for f in fs) / 1e6
print(f"\n{'consumer':22s}{'median':>9s}{'min':>9s}{'max':>9s}{'size':>9s}")
print(f"{'coreml native':22s}{n_med:9.1f}{n_min:9.1f}{n_max:9.1f}{size_native:8.1f}M")
print(f"{'executorch + coreml':22s}{e_med:9.1f}{e_min:9.1f}{e_max:9.1f}"
      f"{os.path.getsize(pte)/1e6:8.1f}M")
print(f"\nexecutorch overhead: {100 * (e_med - n_med) / n_med:+.1f}%")
print(f"outputs agree: corr={np.corrcoef(native_out.ravel(), et_out.ravel())[0,1]:.6f} "
      f"max_abs_diff={np.abs(native_out - et_out).max():.4f}")
