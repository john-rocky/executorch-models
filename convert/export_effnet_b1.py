"""EfficientNet-B1 -> ExecuTorch XNNPACK .pte (harness version of the original smoke test)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from harness import convert_and_gate

m = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2).eval()
convert_and_gate(
    "efficientnet_b1",
    m,
    (torch.randn(1, 3, 240, 240),),
    extra_meta={
        "source": "torchvision efficientnet_b1 IMAGENET1K_V2",
        "license": "BSD-3-Clause",
        "preprocess": "RGB, ImageNet norm, 240x240",
        "outputs": "ImageNet logits [1,1000]",
    },
)
