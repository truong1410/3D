# tnet.py
import torch
import torch.nn as nn
import torchvision.models as models

class TNet(nn.Module):
    """
    Nhận 2 frame liên tiếp ghép kênh (6 channels)
    Output: 6-DOF relative pose (tx, ty, tz, rx, ry, rz)
    """
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)

        # Thay conv1 từ 3 → 6 channels
        base.conv1 = nn.Conv2d(6, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)

        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )
        # Init nhỏ để pose ban đầu gần 0
        nn.init.normal_(self.decoder[-1].weight, 0, 0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, frame_t, frame_t1):
        x = torch.cat([frame_t, frame_t1], dim=1)  # (B, 6, H, W)
        return self.decoder(self.pool(self.encoder(x)))
