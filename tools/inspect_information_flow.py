import argparse
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
from yolox.exp import get_exp
from yolox.models.ttt_modules import DeepSeekSparseAttention

def check_tensor_health(name, tensor):
    if tensor is None: return
    mag = torch.norm(tensor.float(), dim=-1).mean().item()
    std = torch.std(tensor.float()).item()
    zeros = (tensor == 0).float().mean().item() * 100
    nans = torch.isnan(tensor).any().item()
    logger.info(f"[{name:<15}] Mag: {mag:>7.4f} | Std: {std:>7.4f} | Zeros: {zeros:>5.1f}% | NaN: {nans}")

def run_deep_diagnostics(exp_file, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = get_exp(exp_file)
    model = exp.get_model()
    model.eval().to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    
    logger.info(f"\n{'='*70}\nDEEP INFORMATION FLOW DIAGNOSTICS (EPOCH 80)\n{'='*70}")

    val_loader = exp.get_eval_loader(batch_size=4, is_distributed=False, legacy=False)
    imgs, _, _, _ = next(iter(val_loader))
    imgs = imgs.to(device)

    # --- 1. SOBEL EDGE HEALTH (Is TTT reconstructing noise?) ---
    logger.info("\n[1] SOBEL EDGE EXTRACTION QUALITY (On OOD Data):")
    ttt_stage = model.backbone.backbone.dark2
    with torch.no_grad():
        stem_feat = model.backbone.backbone.stem(imgs)
        clean_edges = ttt_stage.sobel_extractor(stem_feat)
        mask = ttt_stage._get_deterministic_sobel_mask(clean_edges)
        
    check_tensor_health("Stem Features", stem_feat)
    check_tensor_health("Sobel Edges", clean_edges)
    logger.info(f"Mask Active Ratio: {mask.mean().item() * 100:.2f}% (Target ~50%)")
    if clean_edges.std().item() < 0.05:
        logger.error("-> CRITICAL: Sobel edges are washed out. TTT is reconstructing flat noise.")

    # --- 2. DSA ATTENTION ENTROPY (Is Global Context Dead?) ---
    logger.info("\n[2] DEEPSEEK SPARSE ATTENTION (DSA) HEALTH:")
    attn_entropies = []
    
    def dsa_hook(module, input, output):
        # We need to recreate the attention map locally to measure its entropy
        x = input[0]
        B, C, H, W = x.shape
        K_tokens = max(1, int(H * W * module.ratio))
        scores = module.indexer(x).view(B, -1) 
        _, topk_indices = torch.topk(scores, K_tokens, dim=1) 
        
        qkv = module.qkv(x).view(B, 3*C, -1) 
        q, k, v = torch.chunk(qkv, 3, dim=1) 
        k_s = torch.gather(k, 2, topk_indices.unsqueeze(1).expand(-1, C, -1)) 
        
        scale = C ** -0.5
        attn = (q.transpose(-2, -1) @ k_s) * scale 
        attn_probs = F.softmax(attn, dim=-1) # [B, HW, K]
        
        # Calculate Shannon Entropy of attention weights
        entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-6), dim=-1).mean().item()
        attn_entropies.append(entropy)
        
    handles = []
    for m in model.modules():
        if isinstance(m, DeepSeekSparseAttention):
            handles.append(m.register_forward_hook(dsa_hook))
            
    with torch.no_grad():
        _ = model(imgs)
        
    for h in handles: h.remove()
    
    for i, ent in enumerate(attn_entropies):
        logger.info(f"DSA Layer {i+1} Entropy: {ent:.4f}")
        if ent < 0.1:
            logger.error(f"-> CRITICAL: Layer {i+1} Attention Collapsed. Looking at a single token.")

    # --- 3. REGRESSION TOWER HEALTH (Did L1 Loss destroy it?) ---
    logger.info("\n[3] HEAD FEATURE HEALTH (Regression vs Classification):")
    
    reg_feats, cls_feats = [], []
    def hook_factory(target_list):
        return lambda m, i, o: target_list.append(o.detach())
        
    h1 = model.head.reg_convs[1].register_forward_hook(hook_factory(reg_feats))
    h2 = model.head.cls_convs[1].register_forward_hook(hook_factory(cls_feats))
    
    with torch.no_grad():
        _ = model(imgs)
        
    h1.remove(); h2.remove()
    
    check_tensor_health("Reg Features", reg_feats[0])
    check_tensor_health("Cls Features", cls_feats[0])
    
    if reg_feats[0].std().item() < (cls_feats[0].std().item() * 0.1):
        logger.error("-> CRITICAL: Regression features have collapsed relative to Classification. L1 Loss likely caused gradient shattering.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--exp_file", type=str, required=True)
    parser.add_argument("-c", "--ckpt", type=str, required=True)
    args = parser.parse_args()
    run_deep_diagnostics(args.exp_file, args.ckpt)