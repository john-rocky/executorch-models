# Bisecting a model that exports but comes out wrong

Written while chasing Grounding DINO, and general enough to reuse. Run them in this order; each
one rules out a layer, and the first that moves is the culprit.

| script | what it rules out |
|---|---|
| `real_gdino.py` | that the model is fine — compares on a real image and on what it detects, not just correlation |
| `bisect_gdino.py` | your own graph surgery, then `torch.export`, then the lowering, in three steps |
| `probe_bb.py` | each half of the vision backbone: feature maps, mask downsample, position embedding |
| `probe_swin.py` | the stock upstream module at several input sizes, with no wrapper around it |
| `probe_msda.py` | one suspect operator on its own, on structured input rather than noise |

The order matters. Establishing that `torch.export` is exact before touching the lowering saves
hours of suspecting your own wrapper.
