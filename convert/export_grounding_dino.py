"""Grounding DINO tiny -> ExecuTorch. Open-vocabulary detection: you name the thing in text.

The model people cite as the one that does not convert. Two pieces of graph surgery make it go,
both recorded in KNOWLEDGE.md: the text mask is lifted out of the graph because it is built with
cummax, and export's runtime assertions are stripped because they are not Core ATen ops. The
deformable attention needs nothing — transformers ships a pure-PyTorch form and its grid_sample
lands on ExecuTorch's own kernels.

Fixed shapes throughout, because Core ML rejects dynamic ones and the text length is the only
genuinely variable input. Padding the prompt to a fixed token count costs nothing at inference.
"""
import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection

CKPT = "IDEA-Research/grounding-dino-tiny"
SIDE = 640
TEXT_LEN = 32
PROMPT = "a person. a car. a bicycle. a dog."

processor = AutoProcessor.from_pretrained(CKPT)
model = GroundingDinoForObjectDetection.from_pretrained(CKPT).eval()

text = processor.tokenizer(PROMPT, padding="max_length", max_length=TEXT_LEN,
                          truncation=True, return_tensors="pt")
pixel_values = torch.randn(1, 3, SIDE, SIDE)
# The text self-attention mask groups the prompt's tokens between its "." separators, and it is
# built with cummax/cummin — neither is in the Core ATen opset. It depends only on input_ids,
# never on the image, so it is lifted out of the graph and passed in instead. The app computes it
# once when the user types a prompt, and the model stays open-vocabulary at runtime.
import transformers.models.grounding_dino.modeling_grounding_dino as gdino

reference_masks, reference_positions = gdino.generate_masks_with_special_tokens_and_transfer_map(
    text["input_ids"])
print("text mask", tuple(reference_masks.shape), "positions", tuple(reference_positions.shape))

example = (pixel_values, text["input_ids"], text["token_type_ids"], text["attention_mask"],
           reference_masks, reference_positions)
print("input_ids", tuple(text["input_ids"].shape), "| tokens", text["input_ids"].shape[1])


class Detector(torch.nn.Module):
    """Returns just the two tensors an app needs: per-query scores over text tokens, and boxes."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._masks = None
        self._positions = None
        gdino.generate_masks_with_special_tokens_and_transfer_map = (
            lambda input_ids: (self._masks, self._positions))

    def forward(self, pixel_values, input_ids, token_type_ids, attention_mask,
                text_masks, position_ids):
        # Set before the model runs, so the traced graph reads these inputs at the point where it
        # used to call cummax.
        self._masks = text_masks
        self._positions = position_ids
        out = self.model(pixel_values=pixel_values, input_ids=input_ids,
                         token_type_ids=token_type_ids, attention_mask=attention_mask,
                         return_dict=True)
        return out.logits, out.pred_boxes


wrapped = Detector(model).eval()
with torch.no_grad():
    logits, boxes = wrapped(*example)
print("eager logits", tuple(logits.shape), "boxes", tuple(boxes.shape))

with torch.no_grad():
    exported = torch.export.export(wrapped, example, strict=False)
print("torch.export: OK")

# torch.export plants runtime assertions (_is_all_true and friends) to guard shapes it could not
# prove statically. Every shape here is fixed, so they are trivially true and are not compute —
# but they are not in the Core ATen opset either, so the graph will not lower with them in it.
removed = 0
for node in reversed(list(exported.graph.nodes)):
    name = str(node.target)
    if node.op == "call_function" and ("_assert" in name or "_is_all_true" in name):
        if not node.users:
            exported.graph.erase_node(node)
            removed += 1
exported.graph.eliminate_dead_code()
exported.graph_module.recompile()
print(f"dropped {removed} runtime assertions")

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from collections import Counter

lowered = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()]).to_executorch()
program = lowered.exported_program()
ops = [n for n in program.graph.nodes if n.op == "call_function"]
delegated = [n for n in ops if "executorch_call_delegate" in str(n.target)]
portable = Counter(str(n.target) for n in ops
                   if "executorch_call_delegate" not in str(n.target)
                   and "alloc" not in str(n.target) and "getitem" not in str(n.target))
print(f"delegate calls {len(delegated)} | ops outside the delegate {sum(portable.values())}")
for name, n in portable.most_common(12):
    print(f"   {n:>5}  {name}")
print("pte MB:", round(len(lowered.buffer) / 1e6, 1))

import os
path = os.path.join(os.path.dirname(__file__), "gdino_probe.pte")
with open(path, "wb") as f:
    f.write(lowered.buffer)
print("wrote", path)
