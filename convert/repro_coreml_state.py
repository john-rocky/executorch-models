"""Two ways to carry a cache through the Core ML delegate, one of which runs.

The Core ML delegate cannot run any model that holds a mutable buffer from OSS ExecuTorch:
coremltools compiles every mutable buffer into a Core ML state, and the OSS runtime does
not bind one, so the model builds, loads, and then fails at execute asking for an
`MLState`. `take_over_mutable_buffer=False` is meant to avoid needing state and cannot,
because declining to take the buffer over does not stop it being a buffer.

This is that failure in twenty lines, next to the same cache written as an input and an
output, which has no buffer to compile into a state and runs. The arithmetic is identical;
only where the cache lives differs.

    python convert/repro_coreml_state.py

Related: pytorch/executorch#21855, apple/coremltools#2826.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import coremltools as ct
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime

SLOTS, WIDTH = 16, 8


class BufferCache(torch.nn.Module):
    """The cache the module owns. What `export_llm` produces."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, SLOTS, WIDTH))

    def forward(self, x, pos):
        self.cache.index_copy_(1, pos, x)
        return self.cache.sum(1)


class CarriedCache(torch.nn.Module):
    """The cache the caller owns: in as an argument, out as a result."""

    def forward(self, x, pos, cache):
        cache = cache.index_copy(1, pos, x)
        return cache.sum(1), cache


def lower(module, example, take_over):
    specs = CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision(ct.precision.FLOAT16.value),
    )
    program = to_edge_transform_and_lower(
        torch.export.export(module.eval(), example),
        partitioner=[CoreMLPartitioner(compile_specs=specs,
                                       take_over_mutable_buffer=take_over)],
    )
    mutated = list(program.exported_program().graph_signature.buffers_to_mutate)
    return program.to_executorch().buffer, mutated


def run(name, buffer, inputs):
    path = f"/tmp/{name}.pte"
    open(path, "wb").write(buffer)
    try:
        method = Runtime.get().load_program(path).load_method("forward")
        out = method.execute(list(inputs))
        print(f"  {name}: RAN, first output {tuple(out[0].shape)}")
    except Exception as error:
        print(f"  {name}: FAILED {str(error).splitlines()[0][:70]}")


def main():
    x = torch.randn(1, 1, WIDTH)
    pos = torch.tensor([0])
    cache = torch.zeros(1, SLOTS, WIDTH)

    for take_over in (True, False):
        buffer, mutated = lower(BufferCache(), (x, pos), take_over)
        print(f"cache as a buffer, take_over_mutable_buffer={take_over} "
              f"(mutated buffers in the edge program: {mutated or 'none'})")
        run(f"buffer_cache_{take_over}", buffer, (x, pos))

    buffer, mutated = lower(CarriedCache(), (x, pos, cache), False)
    print(f"cache carried through the signature "
          f"(mutated buffers in the edge program: {mutated or 'none'})")
    run("carried_cache", buffer, (x, pos, cache))


if __name__ == "__main__":
    main()
