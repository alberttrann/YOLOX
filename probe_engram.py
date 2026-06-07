import torch
import torch.nn.functional as F

ckpt_path = "YOLOX_outputs/tde_yolox_s/epoch_80_ckpt.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")["model"]

print("=== Engram Prototype Separation Analysis ===")
classes = ["Car", "Bus", "Truck", "Person", "Rider", "Bike", "Motor", "Light", "Sign"]

for scale in range(3): # P3, P4, P5
    # Find prototype weights in the state dict
    proto_key = f"head.memory_banks.{scale}.prototypes"
    if proto_key not in ckpt:
        continue
        
    prototypes = ckpt[proto_key] # Shape: [9, 128]
    
    # Normalize to hypersphere
    p_norm = F.normalize(prototypes, p=2, dim=-1)
    
    # Calculate Cosine Similarity Matrix
    sim_matrix = torch.matmul(p_norm, p_norm.t())
    
    print(f"\n--- Scale {scale} (P{scale+3}) ---")
    print("Diagonal (Self-Sim):", torch.diag(sim_matrix).mean().item())
    
    # Mask out diagonal to get cross-class similarity
    mask = ~torch.eye(9, dtype=torch.bool)
    cross_sim = sim_matrix[mask].mean().item()
    max_cross_sim = sim_matrix[mask].max().item()
    
    print(f"Average Cross-Class Similarity: {cross_sim:.4f}")
    print(f"Max Cross-Class Similarity: {max_cross_sim:.4f}")
    
    if cross_sim > 0.8:
        print("🚨 WARNING: SEVERE PROTOTYPE COLLAPSE DETECTED. Classes are indistinguishable.")
    elif cross_sim < 0.1:
        print("✅ Prototypes are well separated and orthogonal.")