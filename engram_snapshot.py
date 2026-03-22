import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
CKPT_PATH = "./YOLOX_outputs/tde_yolox_s/epoch_97_ckpt.pth"
CLASS_NAMES = ["car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign"]

def analyze_engram_memory(ckpt_path, epoch_label="97"):
    print(f"--- Analyzing Engram Prototypes from Epoch {epoch_label} ---")
    
    # 1. Load the checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"]
    
    # 2. Robust Key Search
    # Check for DDP prefix and look for any key containing 'memory_banks' and 'prototypes'
    target_key = None
    possible_keys = [k for k in state_dict.keys() if "memory_banks" in k and "prototypes" in k]
    
    # We want Level 1 (P4 scale). Level 0=P3, 1=P4, 2=P5.
    for k in possible_keys:
        if ".1." in k: 
            target_key = k
            break
            
    if target_key is None:
        print("Error: Could not find memory prototypes. Available keys related to memory:")
        print([k for k in state_dict.keys() if "memory" in k][:10], "... (truncated)")
        return
    
    prototypes = state_dict[target_key] # [9, 128]
    
    # 3. Calculate Cosine Similarity Matrix
    # This measures how 'distinct' each class is from the others
    norms = torch.norm(prototypes, p=2, dim=1, keepdim=True)
    normalized_protos = prototypes / (norms + 1e-6)
    similarity_matrix = torch.mm(normalized_protos, normalized_protos.t()).numpy()

    # 4. Save Raw Data for later comparison with Epoch 40
    np.save(f"engram_protos_epoch_{epoch_label}.npy", prototypes.numpy())

    # 5. Visualization: The "Distinctness Heatmap"
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix, 
        annot=True, 
        fmt=".2f", 
        xticklabels=CLASS_NAMES, 
        yticklabels=CLASS_NAMES,
        cmap="YlGnBu",
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title(f"TDE-YOLOX Engram Identity Matrix (Epoch {epoch_label})\n(1.0 = Perfect Recall, 0.0 = Perfect Distinction)")
    
    save_path = f"engram_heatmap_epoch_{epoch_label}.png"
    plt.savefig(save_path)
    print(f"Snapshot saved: {save_path} and engram_protos_epoch_{epoch_label}.npy")
    plt.show()

if __name__ == "__main__":
    if os.path.exists(CKPT_PATH):
        analyze_engram_memory(CKPT_PATH, "97")
    else:
        print(f"Missing checkpoint at {CKPT_PATH}")