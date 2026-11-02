#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import torch
from yolox.exp import get_exp

EXP_FILE = "exps/default/tde_yolox_s.py"
CKPT_FILE = "YOLOX_outputs/tde_yolox_s/best_ckpt.pth"

def main():
    exp = get_exp(EXP_FILE, None)
    model = exp.get_model()
    model.cuda()
    model.eval()

    if os.path.exists(CKPT_FILE):
        print(f"[*] Loading checkpoint: {CKPT_FILE}")
        ckpt = torch.load(CKPT_FILE, map_location="cuda:0")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        clean_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
        model.load_state_dict(clean_dict, strict=False)
    else:
        print(f"[!] Checkpoint not found: {CKPT_FILE}")
        return

    val_loader = exp.get_eval_loader(batch_size=8, is_distributed=False)
    
    print("\n" + "=" * 80)
    print(" FORENSIC PREDICTION AUDIT ON 1 VALIDATION BATCH")
    print("=" * 80)

    for imgs, _, info_imgs, ids in val_loader:
        imgs = imgs.cuda().float()
        with torch.no_grad():
            outputs = model(imgs)
            
        # outputs shape: [B, total_anchors, 14]
        # [x, y, w, h, obj_prob, cls_prob (9 classes)]
        bboxes = outputs[..., :4]
        obj_probs = outputs[..., 4]
        cls_probs = outputs[..., 5:]
        
        class_conf, class_pred = torch.max(cls_probs, dim=-1)
        composite_scores = obj_probs * class_conf

        print(f"Batch Size:                 {imgs.shape[0]}")
        print(f"Total Anchors per Image:    {outputs.shape[1]}")
        print(f"Objectness Probability:     Min = {obj_probs.min():.6f}, Mean = {obj_probs.mean():.6f}, Max = {obj_probs.max():.6f}")
        print(f"Max Class Probability:      Min = {class_conf.min():.6f}, Mean = {class_conf.mean():.6f}, Max = {class_conf.max():.6f}")
        print(f"Composite Score (obj*cls):  Min = {composite_scores.min():.6f}, Mean = {composite_scores.mean():.6f}, Max = {composite_scores.max():.6f}")
        
        print("\nCandidate Boxes Passing Confidence Thresholds:")
        for thresh in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]:
            count = (composite_scores >= thresh).sum().item()
            print(f"  Score >= {thresh:<6}: {count:>6} boxes across batch ({count / imgs.shape[0]:.1f} per image)")

        print("\nSample Bounding Box Coordinates (First 3 Anchors):")
        for i in range(min(3, outputs.shape[1])):
            box = bboxes[0, i].cpu().tolist()
            print(f"  Anchor {i}: cx={box[0]:.1f}, cy={box[1]:.1f}, w={box[2]:.1f}, h={box[3]:.1f}")

        break
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()