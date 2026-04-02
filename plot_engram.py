import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from yolox.exp import get_exp

def plot_engram_space(ckpt_path, exp_file):
    print("Loading Experiment and Model...")
    exp = get_exp(exp_file)
    model = exp.get_model()
    
    print(f"Loading Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    
    # Extract prototypes (We use scale 2, which is P5, the most semantic scale)
    prototypes = model.get_engram_prototypes()[2] # Shape: [9, 128]
    
    # Classes for BDD100K
    classes = ["car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign"]
    
    print("Applying t-SNE...")
    # We use perplexity=2 because we only have 9 points
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    prototypes_2d = tsne.fit_transform(prototypes)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(prototypes_2d[:, 0], prototypes_2d[:, 1], s=100, c='red')
    
    for i, txt in enumerate(classes):
        plt.annotate(txt, (prototypes_2d[i, 0], prototypes_2d[i, 1]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=12)
        
    plt.title("TDE-YOLOX Engram Latent Space (P5 Scale)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("engram_tsne.png", dpi=300)
    print("Saved plot to engram_tsne.png")

if __name__ == "__main__":
    # Example usage:
    # python plot_engram.py -c YOLOX_outputs/tde_yolox_s/latest_ckpt.pth -f exps/default/tde_yolox_s.py
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", type=str, required=True)
    parser.add_argument("-f", "--exp", type=str, required=True)
    args = parser.parse_args()
    plot_engram_space(args.ckpt, args.exp)