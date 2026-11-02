#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import torch
import torchvision
from yolox.exp import get_exp
from yolox.utils import bboxes_iou, xyxy2xywh

EXP_FILE = "exps/default/tde_yolox_s.py"
CKPT_FILE = "YOLOX_outputs/tde_yolox_s/best_ckpt.pth"

def main():
    exp = get_exp(EXP_FILE, None)
    model = exp.get_model()
    model.cuda().eval()

    print(f"[*] Loading checkpoint: {CKPT_FILE}")
    ckpt = torch.load(CKPT_FILE, map_location="cuda:0")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    clean_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict, strict=False)

    val_loader = exp.get_eval_loader(batch_size=4, is_distributed=False)
    
    print("\n" + "=" * 80)
    print(" GROUND-TRUTH MATCHING & LOCALIZATION AUDIT")
    print("=" * 80)

    for imgs, _, info_imgs, ids in val_loader:
        imgs = imgs.cuda().float()
        with torch.no_grad():
            outputs = model(imgs)

        # Parse predictions for image 0
        img_id = int(ids[0])
        img_h, img_w = int(info_imgs[0][0]), int(info_imgs[1][0])
        scale = min(exp.test_size[0] / float(img_h), exp.test_size[1] / float(img_w))
        
        # Postprocess on Image 0
        pred = outputs[0]  # [12096, 14]
        # Convert cxcywh to xyxy
        box_corner = pred.new(pred.shape)
        box_corner[:, 0] = pred[:, 0] - pred[:, 2] / 2
        box_corner[:, 1] = pred[:, 1] - pred[:, 3] / 2
        box_corner[:, 2] = pred[:, 0] + pred[:, 2] / 2
        box_corner[:, 3] = pred[:, 1] + pred[:, 3] / 2
        pred[:, :4] = box_corner[:, :4]

        class_conf, class_pred = torch.max(pred[:, 5:14], dim=1, keepdim=True)
        obj_conf = pred[:, 4:5]
        scores = obj_conf * class_conf

        mask = (scores >= 0.05).squeeze()
        high_boxes = pred[:, :4][mask]
        high_scores = scores[mask].squeeze()
        high_classes = class_pred[mask].squeeze()

        if high_boxes.numel() == 0:
            print("[!] Zero boxes passed score >= 0.05 on Image 0!")
            break

        # Apply NMS
        nms_keep = torchvision.ops.batched_nms(high_boxes, high_scores, high_classes, 0.60)
        final_boxes = (high_boxes[nms_keep] / scale).cpu()
        final_scores = high_scores[nms_keep].cpu()
        final_classes = high_classes[nms_keep].cpu()

        # Query Ground Truth for Image 0 from COCO API
        coco = val_loader.dataset.coco
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        print(f"Image ID: {img_id} (Native Size: {img_w}x{img_h})")
        print(f"Ground Truth Objects in Image: {len(anns)}")
        print(f"Predicted Boxes after NMS (score >= 0.05): {len(final_boxes)}")

        gt_boxes = []
        gt_classes = []
        for a in anns:
            # COCO bbox: [x, y, w, h] -> convert to xyxy
            gx, gy, gw, gh = a["bbox"]
            gt_boxes.append([gx, gy, gx + gw, gy + gh])
            gt_classes.append(a["category_id"])

        if len(gt_boxes) > 0 and len(final_boxes) > 0:
            gt_tensor = torch.tensor(gt_boxes).float()
            pred_tensor = final_boxes.float()

            ious = bboxes_iou(pred_tensor, gt_tensor, xyxy=True)  # [N_pred, N_gt]
            max_iou_per_pred, matched_gt_idx = ious.max(dim=1)

            print(f"\nTop 5 Highest Scoring Detections on Image 0:")
            print(f"{'Rank':<5} | {'Score':<8} | {'Pred Class':<12} | {'GT Class':<12} | {'Max IoU':<8} | {'Status'}")
            print("-" * 75)
            
            class_ids = val_loader.dataset.class_ids
            cats = coco.cats
            
            for r in range(min(5, len(final_boxes))):
                p_cls_idx = int(final_classes[r].item())
                mapped_cat_id = class_ids[p_cls_idx] if p_cls_idx < len(class_ids) else -1
                pred_name = cats.get(mapped_cat_id, {}).get("name", f"id_{mapped_cat_id}")

                g_idx = matched_gt_idx[r].item()
                true_cat_id = gt_classes[g_idx]
                gt_name = cats.get(true_cat_id, {}).get("name", f"id_{true_cat_id}")
                best_iou = max_iou_per_pred[r].item()

                status = "MATCH (TP)" if (mapped_cat_id == true_cat_id and best_iou >= 0.50) else "MISMATCH / FP"
                print(f"{r+1:<5} | {final_scores[r].item():<8.4f} | {pred_name:<12} | {gt_name:<12} | {best_iou:<8.4f} | {status}")

        break
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()