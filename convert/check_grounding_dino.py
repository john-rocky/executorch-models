"""Parity for the exported Grounding DINO: same inputs through eager and through the .pte."""
import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection
import transformers.models.grounding_dino.modeling_grounding_dino as gdino
from executorch.runtime import Runtime

CKPT, SIDE, TEXT_LEN = "IDEA-Research/grounding-dino-tiny", 640, 32
PROMPT = "a person. a car. a bicycle. a dog."
SP = "/private/tmp/claude-501/-Users-majimadaisuke-code-executorch-convert/13797437-6012-4cee-b4bb-660d4be52578/scratchpad"

processor = AutoProcessor.from_pretrained(CKPT)
model = GroundingDinoForObjectDetection.from_pretrained(CKPT).eval()
text = processor.tokenizer(PROMPT, padding="max_length", max_length=TEXT_LEN,
                           truncation=True, return_tensors="pt")
torch.manual_seed(0)
pixel_values = torch.randn(1, 3, SIDE, SIDE)
masks, positions = gdino.generate_masks_with_special_tokens_and_transfer_map(text["input_ids"])

with torch.no_grad():
    out = model(pixel_values=pixel_values, input_ids=text["input_ids"],
                token_type_ids=text["token_type_ids"], attention_mask=text["attention_mask"],
                return_dict=True)
ref_logits, ref_boxes = out.logits, out.pred_boxes

method = Runtime.get().load_program(f"{SP}/gdino_probe.pte").load_method("forward")
got = method.execute([pixel_values, text["input_ids"], text["token_type_ids"],
                      text["attention_mask"], masks, positions])

# Grounding DINO masks the padded text slots by driving their logits to -inf, so a plain
# difference over the whole tensor is -inf minus -inf. Compare where both sides are finite, and
# judge the boxes on the queries that actually scored — the other 800-odd are noise either way.
both = torch.isfinite(got[0]) & torch.isfinite(ref_logits)
d = (got[0][both] - ref_logits[both]).abs()
corr = torch.corrcoef(torch.stack([got[0][both].float(), ref_logits[both].float()]))[0, 1]
print(f"logits (finite {int(both.sum())}/{both.numel()}): max_abs_diff {d.max():.3e} corr {corr:.6f}")

scores = ref_logits.sigmoid().max(-1).values[0]
top = scores.topk(20).indices
db = (got[1][0][top] - ref_boxes[0][top]).abs()
print(f"boxes (top-20 queries): max_abs_diff {db.max():.3e}")
dball = (got[1] - ref_boxes).abs()
print(f"boxes (all 900):        max_abs_diff {dball.max():.3e}")
print(f"top score eager {float(scores.max()):.4f}  pte {float(got[0].sigmoid().max()):.4f}")
