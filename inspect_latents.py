import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from yolox.exp import get_exp
from yolox.utils import load_ckpt

# --- CONFIGURATION ---
CLASS_NAMES = ["car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign"]
COLORS = sns.color_palette("husl", len(CLASS_NAMES))

def parse_args():
    parser = argparse.ArgumentParser("TDE-YOLOX Latent Explorer")
    parser.add_argument("-f", "--exp_file", type=str, required=True, help="Path to experiment file")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Images to sample")
    parser.add_argument("--scale", type=int, default=1, help="FPN Scale to inspect (0=P3, 1=P4, 2=P5)")
    return parser.parse_args()

def extract_latents_and_prototypes(model, val_loader, scale_idx, device):
    model.eval()
    
    # 1. Extract Prototypes
    memory_bank = model.head.memory_banks[scale_idx]
    
    with torch.no_grad():
        raw_prototypes = memory_bank.prototype_layer.weight.detach().cpu()
        normalized_prototypes = F.normalize(raw_prototypes, p=2, dim=-1)
    
    all_latents = []
    
    print(f"Sampling {val_loader.batch_size} images to extract live latent vectors...")
    
    # 2. Run Forward Pass
    with torch.no_grad():
        for i, (imgs, targets, _, _) in enumerate(val_loader):
            if i > 0: break # Just need one batch
            
            imgs = imgs.to(device)
            targets = targets.to(device)
            B = imgs.shape[0]
            
            # Forward pass through Backbone & Neck
            fpn_outs = model.backbone(imgs, ttt_prob=1.0)
            
            x = fpn_outs[scale_idx]
            stem_out = model.head.stems[scale_idx](x)
            cls_feat = model.head.cls_convs[scale_idx](stem_out)
            
            _, C, H, W = cls_feat.shape
            cls_feat_flat = cls_feat.permute(0, 2, 3, 1).reshape(B, H*W, C)
            
            latent_vec = model.head.latent_projectors[scale_idx](cls_feat_flat)
            normalized_latents = F.normalize(latent_vec, p=2, dim=-1) # [B, HW, 128]
            
            # --- LABEL ASSIGNMENT ---
            model.train()
            _, _, _, _, _, _, _, cls_targets, fg_masks = model.head(fpn_outs, targets, imgs)
            model.eval() 
            
            # --- CRITICAL FIX: RESHAPE FG_MASKS ---
            # fg_masks from YOLOX is a flattened 1D tensor [B * Total_Anchors]
            # We must reshape it to [B, Total_Anchors] to slice by scale
            total_anchors = sum([fpn.shape[-2] * fpn.shape[-1] for fpn in fpn_outs])
            fg_masks_2d = fg_masks.view(B, total_anchors)
            
            # Calculate offsets for THIS scale
            hw_sizes = [fpn.shape[-2] * fpn.shape[-1] for fpn in fpn_outs]
            start_idx = sum(hw_sizes[:scale_idx])
            end_idx = start_idx + hw_sizes[scale_idx]
            
            # Extract mask for THIS scale
            scale_mask = fg_masks_2d[:, start_idx:end_idx] # [B, HW_current_scale]
            
            flat_latents = normalized_latents.reshape(B * H * W, -1)
            flat_mask = scale_mask.reshape(-1)
            
            # Keep only the Foreground (Object) latents for offline analysis
            fg_latents = flat_latents[flat_mask]
            
            if fg_latents.shape[0] > 0:
                all_latents.append(fg_latents.cpu().numpy())
            else:
                print("Warning: No foreground objects found in this sample batch.")

    return normalized_prototypes.numpy(), all_latents

def plot_prototype_similarity(prototypes, epoch_name):
    """
    Plots the Cosine Similarity between the Memory Bank Prototypes.
    If CPL (Contrastive Prototype Learning) is working, the diagonal should be 1.0, 
    and the off-diagonals should trend toward 0.0 (orthogonal) or negative (opposite).
    """
    # Prototypes are already L2 normalized, so dot product is Cosine Similarity
    sim_matrix = np.dot(prototypes, prototypes.T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix, 
        annot=True, 
        fmt=".2f", 
        xticklabels=CLASS_NAMES, 
        yticklabels=CLASS_NAMES,
        cmap="coolwarm", # Blue=Negative/Zero, Red=Positive/1.0
        center=0.0,
        vmin=-1.0, 
        vmax=1.0,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title(f"TDE-YOLOX Engram Prototype Similarity (Epoch {epoch_name})\nTarget: Diagonal=1.0, Off-Diagonal \u2192 0.0")
    plt.tight_layout()
    
    save_path = f"engram_similarity_epoch_{epoch_name}.png"
    plt.savefig(save_path)
    print(f"Saved Prototype Similarity Heatmap to {save_path}")

def plot_latent_space_pca(prototypes, epoch_name):
    """
    Projects the 128D prototypes down to 2D using PCA to visualize their spread.
    If they are clumped together, the bank is dead.
    If they are spread out, the InfoNCE loss is pushing them apart.
    """
    pca = PCA(n_components=2)
    protos_2d = pca.fit_transform(prototypes)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(protos_2d[:, 0], protos_2d[:, 1], c=range(len(CLASS_NAMES)), cmap='tab10', s=200, edgecolors='black')
    
    for i, name in enumerate(CLASS_NAMES):
        plt.annotate(name, (protos_2d[i, 0], protos_2d[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
        
    plt.title(f"PCA of 128D Engram Prototypes (Epoch {epoch_name})\nTarget: Maximize Distance Between Points")
    plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
    plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = f"engram_pca_epoch_{epoch_name}.png"
    plt.savefig(save_path)
    print(f"Saved Prototype PCA Plot to {save_path}")

def main():
    args = parse_args()
    
    # Extract Epoch Number from filename (e.g., 'epoch_20_ckpt.pth' -> '20')
    try:
        epoch_name = os.path.basename(args.ckpt).split('_')[1]
    except:
        epoch_name = "X"

    print(f"--- Forensic Latent Analysis: Epoch {epoch_name} ---")
    
    # 1. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = get_exp(args.exp_file)
    
    # 2. Load Model
    model = exp.get_model()
    model.to(device)
    model.eval()
    
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    
    # 3. Setup DataLoader (Use validation set to see OOD feature extraction)
    val_loader = exp.get_eval_loader(args.batch_size, is_distributed=False, legacy=False)
    
    # 4. Extract Data
    prototypes, live_latents = extract_latents_and_prototypes(model, val_loader, args.scale, device)
    
    # 5. Generate Reports
    plot_prototype_similarity(prototypes, epoch_name)
    plot_latent_space_pca(prototypes, epoch_name)
    
    print("Forensic Analysis Complete.")

if __name__ == "__main__":
    main()