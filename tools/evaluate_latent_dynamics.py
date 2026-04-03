import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse
from yolox.exp import get_exp

def hook_fn(module, input, output, name, storage):
    """Intercepts internal tensors during the forward pass."""
    storage[name] = output.detach().cpu()

def run_forensics(exp_file, ckpt_path, num_batches=5):
    print("Initializing Model and Dataset...")
    exp = get_exp(exp_file)
    model = exp.get_model()
    model.eval()
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.cuda()

    # Get Val Loader (Make sure it points to your Adverse set)
    val_loader = exp.get_eval_loader(batch_size=8, is_distributed=False)
    
    # Storage for intercepted tensors
    tensors = {}
    
    # Register Hooks on the P5 scale (Index 2 in the ModuleLists)
    # We want to intercept the Latent Vector, the Gate, and the Class Output
    handles = []
    head = model.head
    handles.append(head.latent_projectors[2].register_forward_hook(
        lambda m, i, o: hook_fn(m, i, o, 'latent', tensors)))
    handles.append(head.uncertainty_gates[2].register_forward_hook(
        lambda m, i, o: hook_fn(m, i, o, 'gate', tensors)))
    handles.append(head.cls_preds[2].register_forward_hook(
        lambda m, i, o: hook_fn(m, i, o, 'cls_out', tensors)))
    handles.append(head.obj_preds[2].register_forward_hook(
        lambda m, i, o: hook_fn(m, i, o, 'obj_out', tensors)))

    all_latents = []
    all_pseudo_labels = []
    all_gates = []

    print(f"Extracting features from {num_batches} batches...")
    with torch.no_grad():
        for i, (imgs, _, _, _) in enumerate(val_loader):
            if i >= num_batches: break
            imgs = imgs.cuda()
            _ = model(imgs) # Triggers hooks
            
            # Tensors are captured. Now we filter for "Objects"
            # obj_out is [B, 1, H, W], latent is [B, HW, 128]
            obj_scores = torch.sigmoid(tensors['obj_out']).view(imgs.shape[0], -1)
            
            # Find pixels where the model thinks there is an object (Score > 0.3)
            obj_mask = obj_scores > 0.3
            
            if obj_mask.sum() > 0:
                # Extract latents for these object pixels
                valid_latents = tensors['latent'][obj_mask]
                all_latents.append(valid_latents)
                
                # Extract Gate values for these object pixels
                valid_gates = tensors['gate'].view(imgs.shape[0], -1, 1)[obj_mask]
                all_gates.append(valid_gates)
                
                # Get the predicted class to color the t-SNE
                cls_scores = torch.sigmoid(tensors['cls_out']).view(imgs.shape[0], exp.num_classes, -1)
                valid_cls = cls_scores.transpose(1, 2)[obj_mask]
                pseudo_labels = torch.argmax(valid_cls, dim=-1)
                all_pseudo_labels.append(pseudo_labels)

    # Cleanup
    for h in handles: h.remove()

    if len(all_latents) == 0:
        print("No objects detected with confidence > 0.3. Model is too young or collapsed.")
        return

    # Aggregate
    X_features = torch.cat(all_latents, dim=0).numpy()
    Y_labels = torch.cat(all_pseudo_labels, dim=0).numpy()
    Gates = torch.cat(all_gates, dim=0).numpy().flatten()
    
    # Get Prototypes
    prototypes = head.memory_banks[2].prototypes.detach().cpu().numpy()

    print(f"Captured {len(X_features)} valid object features.")
    
    # --- PLOT 1: Feature vs Prototype t-SNE ---
    print("Running t-SNE...")
    # Combine features and prototypes for unified t-SNE space
    combined_data = np.vstack([X_features, prototypes])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    combined_2d = tsne.fit_transform(combined_data)
    
    feat_2d = combined_2d[:-exp.num_classes]
    proto_2d = combined_2d[-exp.num_classes:]
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=Y_labels, cmap='tab10', alpha=0.5, s=10)
    plt.scatter(proto_2d[:, 0], proto_2d[:, 1], c='red', marker='X', s=200, edgecolors='black', label='Prototypes')
    
    classes = ["car", "bus", "truck", "person", "rider", "bike", "motor", "traffic light", "traffic sign"]
    for i, txt in enumerate(classes):
        plt.annotate(txt, (proto_2d[i, 0], proto_2d[i, 1]), fontsize=12, weight='bold')
        
    plt.title("Latent Feature vs. Prototype Alignment (P5 Scale)")
    plt.legend()
    plt.savefig("forensic_tsne_features.png")

    # --- PLOT 2: Uncertainty Gate Histogram ---
    plt.figure(figsize=(8, 6))
    sns.histplot(Gates, bins=50, kde=True, color='purple')
    plt.title("Uncertainty Gate (β) Distribution on Detected Objects")
    plt.xlabel("Gate Value (0 = Conv Only, 1 = Memory Only)")
    plt.ylabel("Frequency")
    plt.savefig("forensic_gate_histogram.png")
    
    print("Forensic analysis complete. Check 'forensic_tsne_features.png' and 'forensic_gate_histogram.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", type=str, required=True)
    parser.add_argument("-f", "--exp", type=str, required=True)
    args = parser.parse_args()
    run_forensics(args.exp, args.ckpt)