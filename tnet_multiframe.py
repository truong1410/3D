import torch
import torch.nn as nn
import torchvision.models as models

class TNetMultiFrame(nn.Module):
    """
    TNet mở rộng 3 frame: (t-1, t, t+1)
    Input: 3 frames ghép kênh → 9 channels
    Output: 2 pose vectors
        - xi_{t-1 -> t}
        - xi_{t -> t+1}
    """
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)

        # 3 frames × 3 channels = 9 input channels
        base.conv1 = nn.Conv2d(9, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)

        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.pool    = nn.AdaptiveAvgPool2d(1)

        # Predict 2 poses cùng lúc: (t-1→t) và (t→t+1)
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 12)  # 6 × 2 poses
        )
        nn.init.normal_(self.decoder[-1].weight, 0, 0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, frame_tm1, frame_t, frame_tp1):
        """
        frame_tm1: frame t-1  (B, 3, H, W)
        frame_t:   frame t    (B, 3, H, W)
        frame_tp1: frame t+1  (B, 3, H, W)
        """
        x   = torch.cat([frame_tm1, frame_t, frame_tp1], dim=1)  # (B, 9, H, W)
        out = self.decoder(self.pool(self.encoder(x)))             # (B, 12)
        return out[:, :6], out[:, 6:]  # pose_{t-1→t}, pose_{t→t+1}
