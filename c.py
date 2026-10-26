import torch

ckpt = torch.load("YOLOX_outputs/tde_yolox_s/latest_ckpt.pth", map_location="cpu")["model"]

for k in sorted(ckpt.keys()):
    if any(m in k for m in ["stem", "aligner", "memory", "manifold", "atmo"]):
        t = ckpt[k].float()
        print(f"{k:55s} | Norm: {t.norm():.5f} | Min: {t.min():.4f} | Max: {t.max():.4f}")