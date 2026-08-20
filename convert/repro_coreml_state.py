"""Minimal repro: CoreMLPartitioner(take_over_mutable_buffer=False) still yields a
state model, and the resulting .pte cannot execute.

A KV cache is a registered buffer that the forward writes into. The flag is meant
to keep such buffers out of Core ML's hands — export_llm hardcodes it to False
with the comment "ExecuTorch does not build CoreML delegate runtime to handle
state when using OSS scripts" — but the mutation lands inside the delegated
partition anyway, so coremltools builds a state model regardless.
"""
import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.runtime import Runtime
import coremltools as ct


class Cache(torch.nn.Module):
    """The shape of every KV cache: a buffer, written at a position, read back."""

    def __init__(self, n=16, d=8):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, n, d))

    def forward(self, x, pos):
        # The pattern Qwen3.5's gated delta-net uses for its recurrent state:
        # in-place mutation of a *slice* of the buffer, not of the whole thing.
        self.cache[:1].mul_(0.9)
        self.cache[:1].copy_(self.cache[:1] + x.unsqueeze(1))
        return self.cache.sum(dim=1) + x


model = Cache().eval()
x = torch.randn(1, 8)
pos = torch.tensor([0])

for ios in (ct.target.iOS17, ct.target.iOS18):
    specs = CoreMLBackend.generate_compile_specs(
        compute_precision=ct.precision.FLOAT16,
        compute_unit=ct.ComputeUnit.ALL,
        minimum_deployment_target=ios,
    )
    part = CoreMLPartitioner(compile_specs=specs, take_over_mutable_buffer=False)
    print(f"\n=== minimum_deployment_target={ios}, "
          f"take_over_mutable_buffer={part.take_over_mutable_buffer} ===")
    ep = torch.export.export(model, (x, pos))
    try:
        et = to_edge_transform_and_lower(ep, partitioner=[part]).to_executorch()
    except Exception as e:
        print(f"  lowering failed: {type(e).__name__}: {str(e)[:150]}")
        continue
    path = f"/tmp/repro_state_{ios}.pte"
    open(path, "wb").write(et.buffer)
    try:
        m = Runtime.get().load_program(path).load_method("forward")
        out = m.execute([x, pos])
        print(f"  executes OK, out {tuple(torch.as_tensor(out[0]).shape)}")
    except Exception as e:
        print(f"  execute failed: {type(e).__name__}: {str(e)[:150]}")
