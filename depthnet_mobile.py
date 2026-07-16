import torch
import torch.nn as nn
import torchvision.models as models

class MobileDepthNet(nn.Module):
    """
    MobileNetV3-Small encoder thực tế:
      e0: features[0]     → 16ch,  H/2,  W/2   (256x320)
      e1: features[1:3]   → 24ch,  H/8,  W/8   (64x80)
      e2: features[3:9]   → 48ch,  H/16, W/16  (32x40)
      e3: features[9:12]  → 96ch,  H/32, W/32  (16x20)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        mobile  = models.mobilenet_v3_small(weights=weights)
        f = list(mobile.features)

        self.enc0 = nn.Sequential(*f[:1])    # 16ch,  H/2
        self.enc1 = nn.Sequential(*f[1:3])   # 24ch,  H/8
        self.enc2 = nn.Sequential(*f[3:9])   # 48ch,  H/16
        self.enc3 = nn.Sequential(*f[9:12])  # 96ch,  H/32

        # up3: H/32 → H/16,  input=96,        output=48
        self.up3 = self._upblock(96,      48)
        # up2: H/16 → H/8,   input=48+48=96,  output=24
        self.up2 = self._upblock(48 + 48, 24)
        # up1: H/8  → H/4,   input=24+24=48,  output=16
        # e1 là 24ch nhưng ở H/8, sau cat với d2(H/8) = 24+24=48
        self.up1 = self._upblock(24 + 24, 16)
        # up_mid: H/4 → H/2, không có skip, để align với e0
        self.up_mid = self._upblock(16, 16)
        # up0: H/2  → H,     input=16+16=32,  output=16
        self.up0 = self._upblock(16 + 16, 16)

        # Output heads
        self.disp3 = nn.Sequential(nn.Conv2d(48, 1, 3, padding=1), nn.Sigmoid())
        self.disp2 = nn.Sequential(nn.Conv2d(24, 1, 3, padding=1), nn.Sigmoid())
        self.disp1 = nn.Sequential(nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())
        self.disp0 = nn.Sequential(nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())

    def _upblock(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e0 = self.enc0(x)   # (B, 16, H/2,  W/2)
        e1 = self.enc1(e0)  # (B, 24, H/8,  W/8)
        e2 = self.enc2(e1)  # (B, 48, H/16, W/16)
        e3 = self.enc3(e2)  # (B, 96, H/32, W/32)

        d3 = self.up3(e3)                          # (B, 48, H/16, W/16)
        d2 = self.up2(torch.cat([d3, e2], dim=1))  # (B, 24, H/8,  W/8)
        d1 = self.up1(torch.cat([d2, e1], dim=1))  # (B, 16, H/4,  W/4)
        dm = self.up_mid(d1)                        # (B, 16, H/2,  W/2)
        d0 = self.up0(torch.cat([dm, e0], dim=1))  # (B, 16, H,    W)

        return {
            ("disp", 0): self.disp0(d0),
            ("disp", 1): self.disp1(d1),
            ("disp", 2): self.disp2(d2),
            ("disp", 3): self.disp3(d3),
        }
