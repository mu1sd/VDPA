🚀 快速开始
import torch
from vdpapkg.models.vdpa import VDPA

model = VDPA(num_classes=2, depth=10, embed_dim=512, num_heads=8)
x = torch.randn(1, 3, 512, 512)
logits = model(x)
print(logits.shape)  # [1, 2]
