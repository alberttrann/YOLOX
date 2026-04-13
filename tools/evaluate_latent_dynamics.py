#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse
import os
from yolox.exp import get_exp

# --- CONFIGURATION ---
CLASS_NAMES = ["car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign"]

def hook_fn(module, input, output, name, storage):
    """Intercepts internal tensors during the forward pass."""
    storage[name] = output.detach().cpu()

def run_forensics(exp_file, ckpt_path, num_batches=5, scale_idx=2):
    print(f"Initializing Model and Dataset for Scale P{scale_idx+3}...")
    exp = get_exp(exp_file)
    model = exp.get_model()
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)

    # Get Val Loader (Points to Adverse set for OOD evaluation)
    val_loader = exp.get_eval_loader(batch_size=8, is_distributed=False, legacy=False)
    
    # Storage for intercepted tensors
    tensors = {}
    
    # Register Hooks on the requested scale (default 2 = P5)
    handles = []
    head = model.head
    handles.append(head.latent_projectors[scale_idx].register_forward_hook(
        lambda m, i, o: hook_fn(m, i, o, 'latent', tensors)))
    handles.append(head.uncertainty_gates[scale_idx].register_forward_hook(
        lambda m, i, o: hook_fn(m, i, o, 'gate', tensors)))
    handles.append(head.cls_preds[scale_idx].register_forward_hook(
        lambda m, i, o: hook_fn(m, i, o, 'cls_out', tensors)))
    handles.append(head.obj_preds[scale_idx].register_forward_hook(
        lambda m, i, o: hook_fn(m, i, o, 'obj_out', tensors)))

    all_latents = []
    all_pseudo_labels = []
    all_gates = []

    print(f"Extracting features from {num_batches} batches...")
    
    # Force TTT to 1.0 to ensure we evaluate the adapted features
    with torch.no_grad():
        for i, (imgs, _, _, _) in enumerate(val_loader):
            if i >= num_batches: break
            imgs = imgs.to(device)
            
            # Run forward pass (triggers hooks)
            _ = model(imgs) 
            
            # --- POST-PROCESSING INTERCEPTED TENSORS ---
            # obj_out is [B, 1, H, W]
            obj_scores = torch.sigmoid(tensors['obj_out']).view(imgs.shape[0], -1)
            
            # Find pixels where the model thinks there is an object (Score > 0.3)
            obj_mask = obj_scores > 0.3
            
            if obj_mask.sum() > 0:
                # 1. L2 Normalize the extracted latents to match the Nuclear Head physics
                # latent is [B, HW, 128]
                raw_latents = tensors['latent']
                norm_latents = F.normalize(raw_latents, p=2, dim=-1)
                valid_latents = norm_latents[obj_mask]
                all_latents.append(valid_latents)
                
                # 2. Extract Gate values 
                # gate is [B, HW, 1]
                valid_gates = tensors['gate'].view(imgs.shape[0], -1, 1)[obj_mask]
                all_gates.append(valid_gates)
                
                # 3. Get Pseudo-labels for t-SNE coloring
                # cls_out is [B, Num_Classes, H, W]
                cls_scores = torch.sigmoid(tensors['cls_out']).view(imgs.shape[0], exp.num_classes, -1)
                valid_cls = cls_scores.transpose(1, 2)[obj_mask]
                pseudo_labels = torch.argmax(valid_cls, dim=-1)
                all_pseudo_labels.append(pseudo_labels)

    # Cleanup Hooks
    for h in handles: h.remove()

    if len(all_latents) == 0:
        print("No objects detected with confidence > 0.3. Model is too young or collapsed.")
        return

    # Aggregate Data
    X_features = torch.cat(all_latents, dim=0).numpy()
    Y_labels = torch.cat(all_pseudo_labels, dim=0).numpy()
    Gates = torch.cat(all_gates, dim=0).numpy().flatten()
    
    # --- CRITICAL FIX: L2 Normalize the Prototypes ---
    raw_prototypes = head.memory_banks[scale_idx].prototypes.detach().cpu()
    norm_prototypes = F.normalize(raw_prototypes, p=2, dim=-1).numpy()

    print(f"Captured {len(X_features)} valid object features.")
    
    try:
        epoch_name = os.path.basename(ckpt_path).split('_')[1]
    except:
        epoch_name = "X"

    # --- PLOT 1: Feature vs Prototype t-SNE ---
    print("Running t-SNE...")
    # Combine features and prototypes for unified t-SNE space
    combined_data = np.vstack([X_features, norm_prototypes])
    
    # Adjust perplexity based on the number of samples
    n_samples = combined_data.shape[0]
    perplexity = min(30, n_samples - 1)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    combined_2d = tsne.fit_transform(combined_data)
    
    feat_2d = combined_2d[:-exp.num_classes]
    proto_2d = combined_2d[-exp.num_classes:]
    
    plt.figure(figsize=(12, 10))
    # Plot features
    scatter = plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=Y_labels, cmap='tab10', alpha=0.5, s=15)
    
    # Plot Prototypes as massive Red X's with black borders
    plt.scatter(proto_2d[:, 0], proto_2d[:, 1], c='red', marker='X', s=300, edgecolors='black', linewidths=2, label='Prototypes')
    
    # Annotate Prototypes
    for i, txt in enumerate(CLASS_NAMES):
        plt.annotate(txt, (proto_2d[i, 0], proto_2d[i, 1]), 
                     xytext=(8, 8), textcoords='offset points', 
                     fontsize=14, weight='bold', 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
    plt.title(f"TDE-YOLOX Latent Manifold vs. Memory Prototypes (Epoch {epoch_name}, Scale P{scale_idx+3})", fontsize=16, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    tsne_save_path = f"forensic_tsne_features_e{epoch_name}.png"
    plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE plot to {tsne_save_path}")

    # --- PLOT 2: Uncertainty Gate Histogram ---
    plt.figure(figsize=(10, 6))
    sns.histplot(Gates, bins=50, kde=True, color='purple', edgecolor='black')
    plt.title(f"Uncertainty Gate (β) Distribution on Detected Objects (Epoch {epoch_name})", fontsize=14, weight='bold')
    plt.xlabel("Gate Value (0.0 = Pure CNN, 1.0 = Pure Memory)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    gate_save_path = f"forensic_gate_histogram_e{epoch_name}.png"
    plt.savefig(gate_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Gate Histogram to {gate_save_path}")
    
    print("\nForensic Dynamics Analysis Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TDE-YOLOX Latent Dynamics Explorer")
    parser.add_argument("-f", "--exp_file", type=str, required=True, help="Path to experiment file")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Images to sample")
    parser.add_argument("--scale", type=int, default=2, help="FPN Scale to inspect (0=P3, 1=P4, 2=P5)")
    args = parser.parse_args()
    
    run_forensics(args.exp_file, args.ckpt, args.batch_size, args.scale)