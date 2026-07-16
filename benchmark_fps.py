import sys, time, torch
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D/monodepth2')
sys.path.insert(0, '/mnt/data1/home/bmestudent01/workspace/nguyenhng/3D')

from networks import ResnetEncoder, DepthDecoder
from depthnet_mobile import MobileDepthNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.randn(1, 3, 512, 640).to(DEVICE)

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def measure_fps(model, n=200):
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
    # Measure
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n):
            _ = model(x)
    torch.cuda.synchronize()
    return n / (time.time() - t0)

# ResNet-18 baseline
enc = ResnetEncoder(18, pretrained=False).to(DEVICE)
dec = DepthDecoder(enc.num_ch_enc).to(DEVICE)

class ResNetDepth(torch.nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc; self.dec = dec
    def forward(self, x):
        return self.dec(self.enc(x))

resnet_model  = ResNetDepth(enc, dec)
mobile_model  = MobileDepthNet(pretrained=False).to(DEVICE)

fps_resnet = measure_fps(resnet_model)
fps_mobile = measure_fps(mobile_model)

print("=" * 45)
print(f"{'Model':<20} {'Params':>8} {'FPS':>8}")
print("-" * 45)
print(f"{'ResNet-18':<20} {count_params(resnet_model):>7.1f}M {fps_resnet:>7.1f}")
print(f"{'MobileNetV3':<20} {count_params(mobile_model):>7.1f}M {fps_mobile:>7.1f}")
print(f"\nSpeedup: {fps_mobile/fps_resnet:.2f}x faster")
print(f"Param reduction: {count_params(resnet_model)/count_params(mobile_model):.2f}x smaller")
print("=" * 45)
