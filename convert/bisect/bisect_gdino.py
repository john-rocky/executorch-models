"""Which step broke it: torch.export (plus my graph surgery), or the lowering to ExecuTorch?

Runs the same real image through three things — the original module, the exported program
executed eagerly, and the lowered .pte — and compares each against the original.
"""
import numpy, torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection
import transformers.models.grounding_dino.modeling_grounding_dino as gdino

CKPT, SIDE, TEXT_LEN = "IDEA-Research/grounding-dino-tiny", 640, 32
PROMPT = "a person. a car. a bicycle. a dog."
IMAGE = "/Users/majimadaisuke/code/etvision-demo/app/src/main/assets/street.jpg"

processor = AutoProcessor.from_pretrained(CKPT)
model = GroundingDinoForObjectDetection.from_pretrained(CKPT).eval()
image = Image.open(IMAGE).convert("RGB").resize((SIDE, SIDE), Image.BILINEAR)
mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)
pixel_values = ((torch.from_numpy(numpy.array(image)).permute(2, 0, 1).float() / 255 - mean)
                / std).unsqueeze(0)
text = processor.tokenizer(PROMPT, padding="max_length", max_length=TEXT_LEN,
                           truncation=True, return_tensors="pt")
masks, positions = gdino.generate_masks_with_special_tokens_and_transfer_map(text["input_ids"])

with torch.no_grad():
    ref = model(pixel_values=pixel_values, input_ids=text["input_ids"],
                token_type_ids=text["token_type_ids"], attention_mask=text["attention_mask"],
                return_dict=True)

real_fn = gdino.generate_masks_with_special_tokens_and_transfer_map


class Detector(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._masks = self._positions = None
        gdino.generate_masks_with_special_tokens_and_transfer_map = (
            lambda input_ids: (self._masks, self._positions))

    def forward(self, pixel_values, input_ids, token_type_ids, attention_mask, m, p):
        self._masks, self._positions = m, p
        out = self.model(pixel_values=pixel_values, input_ids=input_ids,
                         token_type_ids=token_type_ids, attention_mask=attention_mask,
                         return_dict=True)
        return out.logits, out.pred_boxes


wrapped = Detector(model).eval()
example = (pixel_values, text["input_ids"], text["token_type_ids"], text["attention_mask"],
           masks, positions)


def report(name, logits, boxes):
    both = torch.isfinite(logits) & torch.isfinite(ref.logits)
    d = (logits[both] - ref.logits[both]).abs()
    corr = torch.corrcoef(torch.stack([logits[both].float(), ref.logits[both].float()]))[0, 1]
    top = ref.logits.sigmoid().max(-1).values[0].topk(20).indices
    db = (boxes[0][top] - ref.pred_boxes[0][top]).abs().max()
    print(f"{name:<28} logits diff {d.max():.3e} corr {corr:.6f} | boxes(top20) {db:.3e}")


# 1. The wrapper itself, with the mask handed in rather than computed. Isolates the hoist.
with torch.no_grad():
    a, b = wrapped(*example)
report("wrapper (mask hoisted)", a, b)

# 2. The exported program, run eagerly. Isolates torch.export.
with torch.no_grad():
    exported = torch.export.export(wrapped, example, strict=False)
    a, b = exported.module()(*example)
report("torch.export, run eagerly", a, b)

# 3. The same program after the assertions are stripped. Isolates that surgery.
removed = 0
for node in reversed(list(exported.graph.nodes)):
    n = str(node.target)
    if node.op == "call_function" and ("_assert" in n or "_is_all_true" in n) and not node.users:
        exported.graph.erase_node(node)
        removed += 1
exported.graph.eliminate_dead_code()
exported.graph_module.recompile()
with torch.no_grad():
    a, b = exported.module()(*example)
report(f"assertions stripped ({removed})", a, b)

# The export is faithful, so the loss is in lowering or in the kernels. Lower twice — once with
# the XNNPACK partitioner and once with nothing delegated — and see which one moves.
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime
import tempfile, os

def run_lowered(partitioners, label):
    prog = to_edge_transform_and_lower(exported, partitioner=partitioners).to_executorch()
    with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
        f.write(prog.buffer)
        path = f.name
    method = Runtime.get().load_program(path).load_method("forward")
    got = method.execute(list(example))
    report(label, got[0], got[1])
    os.unlink(path)

run_lowered([XnnpackPartitioner()], "lowered + XNNPACK")
run_lowered([], "lowered, portable only")
