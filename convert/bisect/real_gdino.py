"""Real-image parity for the exported Grounding DINO, and a look at what it actually detects."""
import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection
import transformers.models.grounding_dino.modeling_grounding_dino as gdino
from executorch.runtime import Runtime

CKPT, SIDE, TEXT_LEN = "IDEA-Research/grounding-dino-tiny", 640, 32
PROMPT = "a person. a car. a bicycle. a dog."
SP = "/private/tmp/claude-501/-Users-majimadaisuke-code-executorch-convert/13797437-6012-4cee-b4bb-660d4be52578/scratchpad"
IMAGE = "/Users/majimadaisuke/code/etvision-demo/app/src/main/assets/street.jpg"

processor = AutoProcessor.from_pretrained(CKPT)
model = GroundingDinoForObjectDetection.from_pretrained(CKPT).eval()

# Fixed square input: the export has static shapes, so the app resizes to the same square rather
# than using the processor's shortest-edge policy.
image = Image.open(IMAGE).convert("RGB").resize((SIDE, SIDE), Image.BILINEAR)
mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)
pixel_values = ((torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255
                 - mean) / std).unsqueeze(0)

text = processor.tokenizer(PROMPT, padding="max_length", max_length=TEXT_LEN,
                           truncation=True, return_tensors="pt")
masks, positions = gdino.generate_masks_with_special_tokens_and_transfer_map(text["input_ids"])

with torch.no_grad():
    out = model(pixel_values=pixel_values, input_ids=text["input_ids"],
                token_type_ids=text["token_type_ids"], attention_mask=text["attention_mask"],
                return_dict=True)

method = Runtime.get().load_program(f"{SP}/gdino_probe.pte").load_method("forward")
got = method.execute([pixel_values, text["input_ids"], text["token_type_ids"],
                      text["attention_mask"], masks, positions])

both = torch.isfinite(got[0]) & torch.isfinite(out.logits)
d = (got[0][both] - out.logits[both]).abs()
corr = torch.corrcoef(torch.stack([got[0][both].float(), out.logits[both].float()]))[0, 1]
print(f"logits: max_abs_diff {d.max():.3e}  corr {corr:.6f}")

scores = out.logits.sigmoid().max(-1).values[0]
top = scores.topk(20).indices
print(f"boxes (top-20): max_abs_diff {(got[1][0][top] - out.pred_boxes[0][top]).abs().max():.3e}")

# What did it find? Decode the phrase each surviving query points at.
ids = text["input_ids"][0]
for source, name in ((out.logits, "eager"), (got[0], "pte")):
    probs = source.sigmoid()[0]
    keep = (probs.max(-1).values > 0.30).nonzero().flatten()
    found = []
    for q in keep[:8]:
        tok = probs[q].argmax().item()
        found.append(f"{processor.tokenizer.decode([ids[tok]])}={probs[q].max():.2f}")
    print(f"{name:<6} over 0.30: {len(keep)} queries  {' '.join(found)}")
