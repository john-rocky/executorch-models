"""Is the XNNPACK failure specific to Grounding DINO, or does stock Swin do it too?"""
import torch, tempfile, os
from transformers import SwinModel
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values, return_dict=True).last_hidden_state


for ckpt, side in (("microsoft/swin-tiny-patch4-window7-224", 224),
                   ("microsoft/swin-tiny-patch4-window7-224", 640),
                   ("microsoft/swin-tiny-patch4-window7-224", 448),
                   ("microsoft/swin-tiny-patch4-window7-224", 256)):
    model = Wrap(SwinModel.from_pretrained(ckpt)).eval()
    x = (torch.randn(1, 3, side, side),)
    with torch.no_grad():
        ref = model(*x)
        ep = torch.export.export(model, x, strict=False)
    for node in reversed(list(ep.graph.nodes)):
        n = str(node.target)
        if node.op == "call_function" and ("_assert" in n or "_is_all_true" in n) and not node.users:
            ep.graph.erase_node(node)
    ep.graph.eliminate_dead_code(); ep.graph_module.recompile()
    prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
        f.write(prog.buffer); path = f.name
    try:
        got = Runtime.get().load_program(path).load_method("forward").execute(list(x))[0]
        d = (got - ref).abs().max()
        c = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1]
        print(f"{side}px: ran. max_abs_diff {d:.3e} corr {c:.6f}")
    except Exception as e:
        print(f"{side}px: FAILED {type(e).__name__}: {str(e)[:120]}")
    os.unlink(path)
