import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
CKPT_PATH = "./YOLOX_outputs/tde_yolox_s/epoch_46_ckpt.pth"
CLASS_NAMES = ["car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign"]

def analyze_engram_memory(ckpt_path, epoch_label="46"):
    print(f"--- Analyzing Engram Prototypes from Epoch {epoch_label} ---")
    
    # 1. Load the checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"]
    
    # 2. Extract prototypes from the P4 scale (Middle scale, usually most stable)
    # The key in state_dict follows the head.memory_banks structure
    # Level 0 = P3, Level 1 = P4, Level 2 = P5
    target_key = "head.memory_banks.1.prototypes" 
    
    if target_key not in state_dict:
        print(f"Error: Could not find {target_key} in checkpoint.")
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
        analyze_engram_memory(CKPT_PATH, "46")
    else:
        print(f"Missing checkpoint at {CKPT_PATH}")