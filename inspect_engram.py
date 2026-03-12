import numpy as np
import torch
import torch.nn.functional as F

def inspect_prototypes(npy_path):
    print(f"--- Forensic Analysis: {npy_path} ---")
    
    # Load the raw weights [9 classes, 128 dimensions]
    protos = np.load(npy_path)
    protos_tensor = torch.tensor(protos)
    
    num_classes, latent_dim = protos.shape
    print(f"Shape: {num_classes} classes, {latent_dim} latent dimensions.")

    # 1. Magnitude Check (Are they dead?)
    magnitudes = torch.norm(protos_tensor, p=2, dim=1)
    print("\n[1] L2 Magnitudes of Prototypes:")
    for i, mag in enumerate(magnitudes):
        print(f"  Class {i}: {mag.item():.4f}")
        
    avg_mag = magnitudes.mean().item()
    if avg_mag < 1e-4:
        print("  >>> ALARM: Magnitudes are near zero. Prototypes have collapsed.")
    elif avg_mag > 100:
        print("  >>> ALARM: Magnitudes have exploded. Learning rate too high.")
    else:
        print("  >>> PASS: Magnitudes are healthy.")

    # 2. Variance Check (Are they just constant values?)
    variances = torch.var(protos_tensor, dim=1)
    print("\n[2] Internal Variance of Prototypes:")
    print(f"  Average Variance: {variances.mean().item():.4f}")
    if variances.mean().item() < 1e-5:
        print("  >>> ALARM: Prototypes have no internal structure (flat vectors).")

    # 3. Orthogonality Check (Raw Dot Product vs Cosine)
    print("\n[3] Cosine Similarity vs Raw Dot Product:")
    
    # Cosine Similarity (What the Heatmap shows)
    norm_protos = F.normalize(protos_tensor, p=2, dim=1)
    cos_sim = torch.matmul(norm_protos, norm_protos.t())
    
    # Raw Dot Product (Unnormalized interaction)
    raw_dot = torch.matmul(protos_tensor, protos_tensor.t())
    
    # Look at off-diagonal max values
    mask = torch.eye(num_classes, dtype=torch.bool)
    max_off_diag_cos = cos_sim[~mask].max().item()
    min_off_diag_cos = cos_sim[~mask].min().item()
    
    print(f"  Max Off-Diagonal Cosine: {max_off_diag_cos:.4f}")
    print(f"  Min Off-Diagonal Cosine: {min_off_diag_cos:.4f}")
    
    # If the max off-diagonal is literally 0.0000, it means the vectors
    # are STILL exactly in their initialization state (nn.init.orthogonal_).
    if abs(max_off_diag_cos) < 1e-3 and abs(min_off_diag_cos) < 1e-3:
        print("  >>> CRITICAL ALARM: The prototypes HAVE NOT MOVED since initialization.")
        print("  >>> Conclusion: The memory_anchor_loss gradients are dead or disconnected.")
    else:
        print("  >>> PASS: Prototypes have drifted from exact mathematical orthogonality, indicating learning.")

if __name__ == "__main__":
    inspect_prototypes("engram_protos_epoch_46.npy")