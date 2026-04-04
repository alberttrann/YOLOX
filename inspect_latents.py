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

# --- CONFIGURATION ---
CLASS_NAMES = ["car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign"]

def parse_args():
    parser = argparse.ArgumentParser("TDE-YOLOX Latent Explorer")
    parser.add_argument("-f", "--exp_file", type=str, required=True, help="Path to experiment file")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Images to sample")
    parser.add_argument("--scale", type=int, default=2, help="FPN Scale to inspect (0=P3, 1=P4, 2=P5)")
    return parser.parse_args()

def extract_latents_and_prototypes(model, val_loader, scale_idx, device):
    model.eval()
    
    # 1. Extract Prototypes (FIXED: Correct Attribute Name)
    memory_bank = model.head.memory_banks[scale_idx]
    
    with torch.no_grad():
        # In TDE-YOLOX, prototypes is a raw Parameter [num_classes, latent_dim]
        raw_prototypes = memory_bank.prototypes.detach().cpu()
        normalized_prototypes = F.normalize(raw_prototypes, p=2, dim=-1)
    
    all_latents = []
    
    print(f"Sampling {val_loader.batch_size} images to extract live latent vectors...")
    
    # 2. Run Forward Pass
    with torch.no_grad():
        for i, (imgs, targets, _, _) in enumerate(val_loader):
            if i > 0: break # Process one batch
            
            imgs = imgs.to(device)
            targets = targets.to(device)
            B = imgs.shape[0]
            
            # Forward through backbone/neck
            # We pass ttt_prob=1.0 to ensure features are fully adapted for analysis
            fpn_outs = model.backbone(imgs, ttt_prob=1.0)
            
            x = fpn_outs[scale_idx]
            stem_out = model.head.stems[scale_idx](x)
            cls_feat = model.head.cls_convs[scale_idx](stem_out)
            
            _, C, H, W = cls_feat.shape
            cls_feat_flat = cls_feat.permute(0, 2, 3, 1).reshape(B, H*W, C)
            
            # Project to same latent space as prototypes
            latent_vec = model.head.latent_projectors[scale_idx](cls_feat_flat)
            normalized_latents = F.normalize(latent_vec, p=2, dim=-1) 
            
            # --- LABEL ASSIGNMENT (FIXED: Return Signature) ---
            # model.head returns 10 values in our high-fidelity implementation
            model.train() # Temp train mode for SimOTA
            outputs_head = model.head(fpn_outs, targets, imgs)
            # Unpack only what we need from the end of the tuple
            cls_targets = outputs_head[-3] # targets
            fg_masks = outputs_head[-2]    # mask
            model.eval() 
            
            # Shape handling
            total_anchors = sum([fpn.shape[-2] * fpn.shape[-1] for fpn in fpn_outs])
            fg_masks_2d = fg_masks.view(B, total_anchors)
            
            hw_sizes = [fpn.shape[-2] * fpn.shape[-1] for fpn in fpn_outs]
            start_idx = sum(hw_sizes[:scale_idx])
            end_idx = start_idx + hw_sizes[scale_idx]
            
            scale_mask = fg_masks_2d[:, start_idx:end_idx] 
            
            flat_latents = normalized_latents.reshape(B * H * W, -1)
            flat_mask = scale_mask.reshape(-1)
            
            fg_latents = flat_latents[flat_mask]
            
            if fg_latents.shape[0] > 0:
                all_latents.append(fg_latents.cpu().numpy())
            else:
                print("Warning: No foreground objects found in this sample batch.")

    return normalized_prototypes.numpy(), all_latents

def plot_prototype_similarity(prototypes, epoch_name):
    sim_matrix = np.dot(prototypes, prototypes.T)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix, annot=True, fmt=".2f", 
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cmap="coolwarm", center=0.0, vmin=-1.0, vmax=1.0
    )
    plt.title(f"TDE-YOLOX Engram Prototype Similarity (Epoch {epoch_name})")
    plt.tight_layout()
    plt.savefig(f"engram_similarity_epoch_{epoch_name}.png")

def plot_latent_space_pca(prototypes, epoch_name):
    pca = PCA(n_components=2)
    protos_2d = pca.fit_transform(prototypes)
    plt.figure(figsize=(10, 8))
    plt.scatter(protos_2d[:, 0], protos_2d[:, 1], c=range(len(CLASS_NAMES)), cmap='tab10', s=200, edgecolors='black')
    for i, name in enumerate(CLASS_NAMES):
        plt.annotate(name, (protos_2d[i, 0], protos_2d[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
    plt.title(f"PCA of Engram Prototypes (Epoch {epoch_name})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"engram_pca_epoch_{epoch_name}.png")

def main():
    args = parse_args()
    try:
        epoch_name = os.path.basename(args.ckpt).split('_')[1]
    except:
        epoch_name = "X"

    print(f"--- Forensic Latent Analysis: Epoch {epoch_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = get_exp(args.exp_file)
    model = exp.get_model()
    model.to(device)
    
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    
    val_loader = exp.get_eval_loader(args.batch_size, is_distributed=False, legacy=False)
    
    prototypes, _ = extract_latents_and_prototypes(model, val_loader, args.scale, device)
    plot_prototype_similarity(prototypes, epoch_name)
    plot_latent_space_pca(prototypes, epoch_name)
    print("Forensic Analysis Complete.")

if __name__ == "__main__":
    main()