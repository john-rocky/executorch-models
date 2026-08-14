"""Depth-Anything-V2-Small through the Core ML backend instead of XNNPACK.

Everything on the shelf is XNNPACK, which is CPU-only, and the device numbers say
so: this model runs 634 ms on an iPhone 17 Pro. ExecuTorch ships a Core ML backend
in the same package, and Core ML can place work on the Neural Engine. This exports
the same graph through it so the two can be compared on one device.

Compute unit is a compile spec, not a runtime switch — it is baked into the .pte —
so each target gets its own file.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import coremltools as ct
from harness import convert_and_gate
from transformers import AutoModelForDepthEstimation
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner


class DepthWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).predicted_depth


# ALL lets Core ML choose per-op and is what an app would ship; CPU_AND_NE forces
# the question "does this graph actually reach the Neural Engine".
UNITS = {
    "all": ct.ComputeUnit.ALL,
    "ne": ct.ComputeUnit.CPU_AND_NE,
    "gpu": ct.ComputeUnit.CPU_AND_GPU,
}

m = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf")
x = torch.randn(1, 3, 518, 518)
from calib import calib_loader
batches = calib_loader("general", 518, "imagenet")

for name in (sys.argv[1:] or ["all"]):
    specs = CoreMLBackend.generate_compile_specs(
        compute_precision=ct.precision.FLOAT16,
        compute_unit=UNITS[name],
        minimum_deployment_target=ct.target.iOS17,
    )
    convert_and_gate(
        f"depth_anything_v2_small_coreml_{name}",
        DepthWrapper(m), (x,),
        partitioner=CoreMLPartitioner(compile_specs=specs),
        gate_inputs=batches[0],
        extra_meta={
            "source": "depth-anything/Depth-Anything-V2-Small-hf",
            "license": "Apache-2.0",
            "preprocess": "RGB, ImageNet norm, 518x518",
            "outputs": "relative inverse depth [1,518,518]",
            "backend": f"Core ML, fp16 compute, compute_unit={name}",
        },
    )
