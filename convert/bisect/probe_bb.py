"""Inside Grounding DINO's backbone wrapper: the feature-map path, or the mask downsample?"""
import torch, tempfile, os
from transformers import GroundingDinoForObjectDetection
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime

SIDE = 640
model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").eval()
conv = model.model.backbone.conv_encoder if hasattr(model.model.backbone, "conv_encoder") else model.model.backbone
print("wrapper type:", type(conv).__name__, "| inner:", type(conv.model).__name__)


def check(label, module, example):
    module = module.eval()
    with torch.no_grad():
        ref = module(*example)
    refs = [ref] if isinstance(ref, torch.Tensor) else list(ref)
    try:
        with torch.no_grad():
            ep = torch.export.export(module, example, strict=False)
        for n in reversed(list(ep.graph.nodes)):
            t = str(n.target)
            if n.op == "call_function" and ("_assert" in t or "_is_all_true" in t) and not n.users:
                ep.graph.erase_node(n)
        ep.graph.eliminate_dead_code(); ep.graph_module.recompile()
        prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
        with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
            f.write(prog.buffer); p = f.name
        got = Runtime.get().load_program(p).load_method("forward").execute(list(example))
        os.unlink(p)
        worst = max(float((a.float() - b.float()).abs().max()) for a, b in zip(got, refs))
        print(f"{label:<34} OK   max_abs_diff {worst:.3e}")
    except Exception as e:
        print(f"{label:<34} FAIL {type(e).__name__}: {str(e)[:100]}")


class FeatureMaps(torch.nn.Module):
    """The multi-level feature maps only — no mask arithmetic."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
    def forward(self, pixel_values):
        return tuple(self.inner(pixel_values, return_dict=True).feature_maps)


class MaskOnly(torch.nn.Module):
    """The mask downsample only — interpolate to each level, cast to bool."""
    def forward(self, pixel_mask):
        out = []
        for size in ((160, 160), (80, 80), (40, 40)):
            out.append(torch.nn.functional.interpolate(pixel_mask[None].float(), size=size)
                       .to(torch.bool)[0])
        return tuple(out)


class PositionEmbedding(torch.nn.Module):
    """The sine position embedding, built from the downsampled mask with cumsum."""
    def __init__(self, m):
        super().__init__()
        self.pos = m.model.backbone.position_embedding
    def forward(self, feature_map, mask):
        return self.pos(feature_map, mask)


class WholeBackbone(torch.nn.Module):
    """Feature maps, masks and position embeddings together — the module that failed."""
    def __init__(self, m):
        super().__init__()
        self.b = m.model.backbone
    def forward(self, pixel_values, pixel_mask):
        out, pos = self.b(pixel_values, pixel_mask)
        return tuple(f for f, _ in out) + tuple(pos)


class MaskAsConstant(torch.nn.Module):
    """Identical to WholeBackbone except the mask is built inside forward, so it is baked into the
    graph as a constant rather than arriving as an input."""
    def __init__(self, m):
        super().__init__()
        self.b = m.model.backbone
    def forward(self, pixel_values):
        out, pos = self.b(pixel_values, torch.ones(1, SIDE, SIDE, dtype=torch.long))
        return tuple(f for f, _ in out) + tuple(pos)


check("mask as graph constant", MaskAsConstant(model), (torch.randn(1, 3, SIDE, SIDE),))
check("backbone feature_maps", FeatureMaps(conv.model), (torch.randn(1, 3, SIDE, SIDE),))
check("mask downsample", MaskOnly(), (torch.ones(1, SIDE, SIDE, dtype=torch.long),))
check("sine position embedding", PositionEmbedding(model),
      (torch.randn(1, 192, 160, 160), torch.ones(1, 160, 160, dtype=torch.bool)))
check("whole backbone (the failure)", WholeBackbone(model),
      (torch.randn(1, 3, SIDE, SIDE), torch.ones(1, SIDE, SIDE, dtype=torch.long)))
