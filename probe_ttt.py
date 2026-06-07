import torch
from exps.default.test_acdc_snow import Exp

def test_ttt_telemetry():
    print("=== TTT Functional Telemetry Probe ===")
    exp = Exp()
    model = exp.get_model()
    
    ckpt = torch.load("YOLOX_outputs/tde_yolox_s/epoch_80_ckpt.pth", map_location="cuda:0")
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    
    dummy_img = torch.randn(1, 3, 640, 640).cuda()
    
    # 1. Run backbone WITHOUT TTT
    with torch.no_grad():
        # Corrected Path: Must pass through STEM first
        x_stem = model.backbone.backbone.stem(dummy_img)
        feat_static = model.backbone.backbone.dark2(x_stem, run_ttt=False)
    
    # 2. Run backbone WITH TTT
    with torch.enable_grad():
        # Corrected Path
        x_stem_adapt = model.backbone.backbone.stem(dummy_img).detach().requires_grad_(True)
        feat_adapted = model.backbone.backbone.dark2(x_stem_adapt, run_ttt=True)
    
    # 3. Measure Feature Shift
    feature_diff = torch.norm(feat_adapted - feat_static).item()
    print(f"\nFeature Map Shift (L2 Norm): {feature_diff:.4f}")
    
    if feature_diff == 0.0:
        print("🚨 TTT Failed: The output features didn't change. Functional update failed.")
    elif feature_diff > 0.0 and feature_diff < 500.0:
        print("✅ TTT Success: Features successfully adapted to the current image.")
    else:
        print("⚠️ TTT Warning: Massive feature shift, learning rate might be too high.")

if __name__ == "__main__":
    test_ttt_telemetry()