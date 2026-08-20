"""Does Grounding DINO's deformable attention survive export and lowering?

This is the op the model is famous for not converting: everywhere else it is a custom CUDA
kernel. Transformers ships a pure-PyTorch form of it, so the question is only whether that form
lowers — and it is worth answering before spending an hour on 690 MB of weights.
"""
import torch
from transformers.models.grounding_dino.modeling_grounding_dino import MultiScaleDeformableAttention
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# Grounding DINO tiny: 8 heads x 32 dims, 4 feature levels, 4 sampling points, 900 queries.
B, HEADS, DIM, QUERIES, LEVELS, POINTS = 1, 8, 32, 900, 4, 4
shapes = [(100, 134), (50, 67), (25, 34), (13, 17)]
total = sum(h * w for h, w in shapes)

# Random values hide a sampling bug: sample the wrong pixel of white noise and the answer still
# looks statistically right. Structured features do not forgive that, so the value tensor is a
# smooth ramp per level — the same thing a real backbone would hand over.
rows = []
for h, w in shapes:
    yy = torch.linspace(0, 1, h).view(h, 1).expand(h, w)
    xx = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    rows.append((yy * 3 + xx * 5).reshape(1, h * w, 1, 1).expand(B, h * w, HEADS, DIM))
value = torch.cat(rows, dim=1).contiguous()
spatial = torch.tensor(shapes, dtype=torch.long)
start = torch.tensor([0] + list(torch.cumsum(torch.tensor([h * w for h, w in shapes]), 0)[:-1]))
locations = torch.rand(B, QUERIES, HEADS, LEVELS, POINTS, 2)
weights = torch.rand(B, QUERIES, HEADS, LEVELS, POINTS)


class Wrapped(torch.nn.Module):
    """The list-of-tuples argument is Python-level control flow, so it is closed over rather than
    passed as an input; the spatial shapes are fixed once the input resolution is."""

    def __init__(self):
        super().__init__()
        self.attn = MultiScaleDeformableAttention()

    def forward(self, value, level_start_index, sampling_locations, attention_weights):
        return self.attn(value, spatial, shapes, level_start_index,
                         sampling_locations, attention_weights, 64)


model = Wrapped().eval()
example = (value, start, locations, weights)
with torch.no_grad():
    reference = model(*example)
print("eager output", tuple(reference.shape))

exported = torch.export.export(model, example, strict=False)
print("torch.export: OK")

lowered = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()]).to_executorch()
program = lowered.exported_program()
from collections import Counter
ops = [n for n in program.graph.nodes if n.op == "call_function"]
delegated = [n for n in ops if "executorch_call_delegate" in str(n.target)]
portable = Counter(str(n.target) for n in ops if "executorch_call_delegate" not in str(n.target))
print(f"lowered: {len(ops)} call_function nodes, {len(delegated)} delegate calls")
print(f"ops outside the delegate: {sum(portable.values())}")
for name, n in portable.most_common(10):
    print(f"   {n:>4}  {name}")
print("pte bytes:", len(lowered.buffer))

# Lowering is not correctness. Run the program and compare against eager.
import tempfile, os
from executorch.runtime import Runtime

with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
    f.write(lowered.buffer)
    path = f.name
method = Runtime.get().load_program(path).load_method("forward")
got = method.execute(list(example))[0]
diff = (got - reference).abs()
corr = torch.corrcoef(torch.stack([got.flatten(), reference.flatten()]))[0, 1]
print(f"parity vs eager: max_abs_diff {diff.max():.3e}  corr {corr:.6f}")
os.unlink(path)
