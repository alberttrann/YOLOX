import torch
ckpt = torch.load("YOLOX_outputs/tde_yolox_s/best_ckpt.pth", map_location="cpu")
print(f"--- BEST MODEL IDENTITY ---")
print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"AP:    {ckpt.get('best_ap', 0.0):.4f}")
print(ckpt.keys())
print(f"--- MODEL STATE DICT KEYS ---")
for k in ckpt["model"].keys():
    print(k)
print(f"--- OPTIMIZER STATE DICT KEYS ---")
if 'optimizer' in ckpt:
    print(ckpt['optimizer'].keys())
if 'best_ap' in ckpt:
    print(f"Best AP: {ckpt['best_ap']}")
if 'curr_ap' in ckpt:
    print(f"Current AP: {ckpt['curr_ap']}")
