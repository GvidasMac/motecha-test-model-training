import torch
import torch.nn as nn
from torchvision import models

class MobileNetV3Backbone(nn.Module):
    """
    Minimal MobileNetV3-Small backbone truncated to stride=8 feature map.
    Output: [B, C, H/8, W/8]
    """
    def __init__(self, out_indices=5):
        super().__init__()
        m = models.mobilenet_v3_small(weights=None)
        # features is a Sequential; up to index 6 gives stride 8, but 0..5 inclusive
        # we’ll slice to :6 to be safe; adjust if you want different channels
        self.stem = nn.Sequential(*list(m.features[:6]))
        # Infer out channels by a dummy pass or hardcode ~48 (typical for small@block5)
        self.out_channels = list(m.features[:6])[-1].out_channels if hasattr(list(m.features[:6])[-1], "out_channels") else 48

    def forward(self, x):
        return self.stem(x)