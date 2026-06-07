import argparse
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
from yolox.exp import get_exp

def run_forensics(exp_file, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp = get_exp(exp_file)
    model = exp.get_model()
    model.eval().to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    
    logger.info(f"\n{'='*60}\nEPOCH 80 FORENSIC AUTOPSY REPORT\n{'='*60}")

    # --- 1. TTT-LR EXPLOSION CHECK ---
    ttt_stage = model.backbone.backbone.dark2
    lrs = [F.softplus(p).item() for p in ttt_stage.ttt_lrs.values()]
    avg_lr = np.mean(lrs)
    max_lr = np.max(lrs)
    logger.info("\n[1] TTT META-LR HEALTH:")
    logger.info(f"Average Active LR: {avg_lr:.6f} | Max LR: {max_lr:.6f}")
    if avg_lr > 0.5:
        logger.error("-> FAILURE: TTT LR exploded. The inner loop destroyed the Norm layers.")
    elif avg_lr < 0.001:
        logger.warning("-> FAILURE: TTT LR vanished. The model stopped adapting.")
    else:
        logger.success("-> PASS: TTT LR is in a healthy range.")

    # --- 2. ENGRAM PROTOTYPE SHATTERING CHECK ---
    banks = model.head.memory_banks
    sim_scores = []
    for bank in banks:
        p_norm = F.normalize(bank.prototypes, p=2, dim=-1)
        sim_matrix = torch.matmul(p_norm, p_norm.t())
        mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=device)
        sim_scores.append(sim_matrix[mask].mean().item())
    
    avg_sim = np.mean(sim_scores)
    logger.info(f"\n[2] MEMORY BANK ORTHOGONALITY:")
    logger.info(f"Mean Off-Diagonal Similarity: {avg_sim:.4f}")
    if avg_sim < -0.2:
        logger.error("-> FAILURE: Prototypes repelled too hard (Shattered Space).")
    elif avg_sim > 0.5:
        logger.error("-> FAILURE: Prototypes collapsed into a single point.")
    else:
        logger.success("-> PASS: Prototypes are healthily distributed.")

    # --- 3. CONVEX GATE ERASURE CHECK ---
    logger.info("\n[3] ENGRAM GATE ERASURE CHECK (Running Inference...)")
    val_loader = exp.get_eval_loader(batch_size=4, is_distributed=False, legacy=False)
    
    gate_values = {0: [], 1: [], 2: []}
    
    def get_gate_hook(scale_idx):
        return lambda m, i, o: gate_values[scale_idx].append(o.detach())
        
    handles = [
        model.head.uncertainty_gates[k].register_forward_hook(get_gate_hook(k))
        for k in range(3)
    ]
    
    with torch.no_grad():
        imgs, _, _, _ = next(iter(val_loader))
        _ = model(imgs.to(device))
        
    for h in handles: h.remove()
    
    for k in range(3):
        g_mean = torch.cat(gate_values[k]).mean().item()
        g_max = torch.cat(gate_values[k]).max().item()
        logger.info(f"Scale P{k+3} Gate -> Mean: {g_mean:.4f} | Max: {g_max:.4f}")
        if g_mean > 0.7:
            logger.error(f"-> FAILURE: Gate is locked OPEN. Erasing {g_mean*100:.1f}% of spatial Conv features!")
        elif g_mean < 0.05:
            logger.warning(f"-> FAILURE: Gate is locked CLOSED. Engram is dead.")
            
    logger.info("\n" + "="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--exp_file", type=str, required=True)
    parser.add_argument("-c", "--ckpt", type=str, required=True)
    args = parser.parse_args()
    run_forensics(args.exp_file, args.ckpt)