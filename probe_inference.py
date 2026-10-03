#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Integrated for TDE-YOLOX v3.2: Cross-Domain Adverse Weather Forensic Probe

import os
import sys
import math
import time
import argparse
from datetime import datetime

# === WINDOWS UTF-8 ENCODING SHIELD ===
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
# =====================================

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.exp import get_exp
from yolox.utils import postprocess, load_ckpt
from yolox.data.data_augment import ValTransform

class Style:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

CATEGORY_MAP = {
    1: "car", 2: "bus", 3: "truck", 4: "person", 
    5: "rider", 6: "bike", 7: "motor", 8: "traffic light", 9: "traffic sign"
}
CLASS_COLORS = [
    (0.2, 0.4, 0.8), (0.8, 0.2, 0.2), (0.9, 0.5, 0.1), (0.2, 0.8, 0.2),
    (0.6, 0.2, 0.8), (0.2, 0.8, 0.8), (0.8, 0.8, 0.2), (0.9, 0.1, 0.5), (0.5, 0.5, 0.5)
]


class DualLogger:
    """Tees output to terminal and a log file, stripping ANSI for disk storage."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self.log_file = open(log_path, "a", encoding="utf-8", errors="replace")
        self.ansi_cleaner = None
        try:
            import re
            self.ansi_cleaner = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        except Exception:
            pass

    def write(self, message):
        self.terminal.write(message)
        clean_msg = self.ansi_cleaner.sub('', message) if self.ansi_cleaner else message
        self.log_file.write(clean_msg)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if self.log_file and not self.log_file.closed:
            self.log_file.close()


class TDEInferenceXRay:
    """
    Diagnostic harness intercepting intermediate tensors across the model.
    """
    def __init__(self, model):
        self.model = model
        self.taps = {}
        self._install_probes()

    def _install_probes(self):
        m = self.model.module if hasattr(self.model, "module") else self.model

        # TAP 1: P5 Optical Entropy and Saliency
        if hasattr(m.backbone, "p5_dense_attn"):
            orig_p5_fwd = m.backbone.p5_dense_attn.forward
            def tapped_p5_fwd(x):
                p5_feat, saliency, h_s = orig_p5_fwd(x)
                self.taps["p5_saliency"] = saliency.detach().float().cpu()
                self.taps["p5_entropy_raw"] = h_s.item() if hasattr(h_s, "item") else float(h_s)
                return p5_feat, saliency, h_s
            m.backbone.p5_dense_attn.forward = tapped_p5_fwd

        # TAP 2: P4 BQSA Block Selection & Adaptive Tau
        if hasattr(m.backbone, "p4_bqsa"):
            orig_p4_fwd = m.backbone.p4_bqsa.forward
            def tapped_p4_fwd(x, p5_saliency, h_s, targets=None):
                p4_out, topk_idx, indexer_loss = orig_p4_fwd(x, p5_saliency, h_s, targets=targets)
                self.taps["p4_topk_indices"] = topk_idx.detach().cpu()
                h_ratio = torch.clamp(h_s.float() / math.log(max(2, 100)), 0.0, 1.0)
                tau_adaptive = torch.clamp(m.backbone.p4_bqsa.tau * (1.0 - 0.35 * h_ratio), min=0.10)
                self.taps["tau_adaptive"] = tau_adaptive.item() if hasattr(tau_adaptive, "item") else float(tau_adaptive)
                return p4_out, topk_idx, indexer_loss
            m.backbone.p4_bqsa.forward = tapped_p4_fwd

        # TAP 3: Scale-AttnRes Highway Mixing Weights
        for name in ["attnres_n3", "attnres_n4", "attnres_n5"]:
            if hasattr(m.backbone, name):
                res_module = getattr(m.backbone, name)
                orig_res_fwd = res_module.forward
                def make_tapped_res(orig_fn, res_name, res_mod):
                    def tapped(sources):
                        with torch.no_grad():
                            V = torch.stack(sources, dim=0)
                            K = torch.stack([res_mod.norm(s) for s in sources], dim=0)
                            w_safe = torch.clamp(res_mod.w, min=-5.0, max=5.0)
                            logits = torch.einsum('c, m b c h w -> m b h w', w_safe, K) * res_mod.scale
                            weights = F.softmax(logits.float(), dim=0)
                            self.taps[f"{res_name}_weights"] = weights.mean(dim=[1, 2, 3]).detach().cpu().numpy()
                        return orig_fn(sources)
                    return tapped
                res_module.forward = make_tapped_res(orig_res_fwd, name, res_module)

        # TAP 4: Stage 1 TTT Inner Loop & Weight Shifts
        if hasattr(m.backbone.backbone, "dark2"):
            ttt_stage = m.backbone.backbone.dark2
            orig_ttt_fwd = ttt_stage.forward
            def tapped_ttt_fwd(x_in, run_ttt=True):
                # In probe mode, execute with TTT tracking
                out, aux_loss = orig_ttt_fwd(x_in, run_ttt=run_ttt)
                self.taps["ttt_executed"] = run_ttt
                
                # Compute inner adaptation loss dynamically for evaluation
                with torch.no_grad():
                    clean_target = ttt_stage.backbone_stage(x_in).detach()
                    mask = ttt_stage._get_robust_contrast_mask(clean_target)
                    self.taps["ttt_contrast_mask"] = mask.detach().float().cpu()
                    self.taps["ttt_mask_ratio"] = mask.float().mean().item()
                    
                    # Live feature reconstruction error
                    token = ttt_stage.projector.mask_token.to(clean_target.dtype)
                    corrupted = clean_target * (1.0 - mask) + token * mask
                    rec = ttt_stage.projector(corrupted)
                    live_loss = F.mse_loss(rec.float(), clean_target.float()).item()
                    self.taps["ttt_inner_loss"] = live_loss
                    
                return out, aux_loss
            ttt_stage.forward = tapped_ttt_fwd

        # TAP 5: Engram Head Gating & Memory Retrieval
        if hasattr(m, "head") and hasattr(m.head, "memory_banks"):
            for idx, bank in enumerate(m.head.memory_banks):
                orig_bank_fwd = bank.forward
                def make_tapped_bank(orig_fn, bank_idx):
                    def tapped(anchor_feat, h_local, objectness_mask, z_conv_logits, epoch=0, h_s=0.0):
                        mem_spatial, gate_spatial, raw_scores = orig_fn(
                            anchor_feat, h_local, objectness_mask, z_conv_logits, epoch=epoch, h_s=h_s
                        )
                        self.taps[f"engram_scale_{bank_idx}_gate"] = gate_spatial.detach().float().cpu()
                        self.taps[f"engram_scale_{bank_idx}_mean_gate"] = gate_spatial.mean().item()
                        self.taps[f"engram_scale_{bank_idx}_active_pixels"] = (gate_spatial > 0.10).float().mean().item() * 100.0
                        return mem_spatial, gate_spatial, raw_scores
                    return tapped
                bank.forward = make_tapped_bank(orig_bank_fwd, idx)

    def clear(self):
        self.taps = {}


def run_counterfactual_ablation(model, x, exp):
    """Measures attribution by counterfactually turning components off."""
    m = model.module if hasattr(model, "module") else model
    results = {}

    with torch.no_grad():
        # Baseline: Full TDE-YOLOX
        out_full = m(x)
        dets_full = postprocess(out_full, exp.num_classes, exp.test_conf, exp.nmsthre)[0]
        results["Full"] = dets_full if dets_full is not None else torch.zeros((0, 7))

        # Ablation 1: Turn off Stage 1 TTT Adaptation
        orig_dark2 = m.backbone.backbone.dark2.forward
        m.backbone.backbone.dark2.forward = lambda x_in, run_ttt=True: (m.backbone.backbone.dark2.backbone_stage(x_in), torch.tensor(0.0, device=x.device))
        out_no_ttt = m(x)
        dets_no_ttt = postprocess(out_no_ttt, exp.num_classes, exp.test_conf, exp.nmsthre)[0]
        results["No_TTT"] = dets_no_ttt if dets_no_ttt is not None else torch.zeros((0, 7))
        m.backbone.backbone.dark2.forward = orig_dark2

        # Ablation 2: Turn off Engram Memory Bank
        orig_bank_fwds = [b.forward for b in m.head.memory_banks]
        for b in m.head.memory_banks:
            def make_dummy_bank_fwd(target_dim):
                return lambda a, h, obj, z, epoch=0, h_s=0.0: (
                    torch.zeros((h.shape[0], target_dim, h.shape[2], h.shape[3]), device=h.device, dtype=h.dtype),
                    torch.zeros((h.shape[0], 1, h.shape[2], h.shape[3]), device=h.device, dtype=h.dtype),
                    torch.zeros((h.shape[0], h.shape[2] * h.shape[3], 9), device=h.device, dtype=h.dtype)
                )
            b.forward = make_dummy_bank_fwd(b.latent_dim)

        out_no_engram = m(x)
        dets_no_engram = postprocess(out_no_engram, exp.num_classes, exp.test_conf, exp.nmsthre)[0]
        results["No_Engram"] = dets_no_engram if dets_no_engram is not None else torch.zeros((0, 7))
        for b, orig_f in zip(m.head.memory_banks, orig_bank_fwds):
            b.forward = orig_f

    return results


def render_diagnostic_dashboard(img_rgb, boxes, taps, ablations, domain_tag, save_path):
    fig = plt.figure(figsize=(26, 16))
    fig.patch.set_facecolor('#121212')

    # Panel 1: Live Detections
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor('#121212')
    ax1.imshow(img_rgb)
    ax1.set_title(f"Live Detections [{domain_tag.upper()}] (Found: {len(boxes)})", color='white', fontsize=14, fontweight='bold')
    ax1.axis('off')

    for box in boxes:
        x1, y1, x2, y2, obj_conf, cls_conf, cls_id = box[:7]
        cls_id = int(cls_id)
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        label = f"{CATEGORY_MAP.get(cls_id + 1, str(cls_id))}: {obj_conf * cls_conf:.2f}"
        ax1.text(x1, y1 - 4, label, color='white', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle="square,pad=0.2", facecolor=color, alpha=0.8, edgecolor='none'))

    # Panel 2: Stage 1 TTT Mask
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor('#121212')
    if "ttt_contrast_mask" in taps:
        mask_np = taps["ttt_contrast_mask"][0, 0].numpy()
        mask_resized = cv2.resize(mask_np, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        ax2.imshow(img_rgb, alpha=0.5)
        ax2.imshow(mask_resized, cmap='magma', alpha=0.7)
        ax2.set_title(f"Stage 1 TTT Saliency Mask (Active: {taps.get('ttt_mask_ratio', 0)*100:.1f}%)", color='white', fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "TTT Mask Inactive", color='gray', ha='center', va='center')
    ax2.axis('off')

    # Panel 3: P5 Optical Saliency
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor('#121212')
    if "p5_saliency" in taps:
        sal_np = taps["p5_saliency"][0, 0].numpy()
        sal_resized = cv2.resize(sal_np, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        ax3.imshow(sal_resized, cmap='plasma')
        h_s = taps.get("p5_entropy_raw", 0.0)
        h_max = math.log(400.0)
        ax3.set_title(f"P5 Optical Attention Saliency (H(s)={h_s:.2f} / {h_max:.2f} nats)", color='white', fontsize=14, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, "P5 Attention Inactive", color='gray', ha='center', va='center')
    ax3.axis('off')

    # Panel 4: Engram Memory Gate
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor('#121212')
    if "engram_scale_0_gate" in taps:
        gate_np = taps["engram_scale_0_gate"][0, 0].numpy()
        gate_resized = cv2.resize(gate_np, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        ax4.imshow(img_rgb, alpha=0.4)
        im4 = ax4.imshow(gate_resized, cmap='inferno', alpha=0.75, vmin=0.0, vmax=0.70)
        plt.colorbar(im4, ax=ax4, fraction=0.035, pad=0.04)
        active_pct = taps.get("engram_scale_0_active_pixels", 0.0)
        ax4.set_title(f"Engram Memory Injection Gate (Active: {active_pct:.1f}%)", color='white', fontsize=14, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, "Memory Gate Inactive", color='gray', ha='center', va='center')
    ax4.axis('off')

    # Panel 5: Scale-AttnRes Highway Flow
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor('#1e1e1e')
    scales = ['N3 (Stride 8)', 'N4 (Stride 16)', 'N5 (Stride 32)']
    data_bars = []
    for name in ["attnres_n3", "attnres_n4", "attnres_n5"]:
        w = taps.get(f"{name}_weights", None)
        data_bars.append(w if w is not None else [1.0 / 3.0] * 3)

    x_indices = np.arange(len(scales))
    width = 0.25

    s0 = [d[0] for d in data_bars]
    s1 = [d[1] for d in data_bars]
    s2 = [d[2] if len(d) > 2 else 0.0 for d in data_bars]

    ax5.bar(x_indices - width, s0, width, label='Top-Down BQSA Context', color='#3498db')
    ax5.bar(x_indices, s1, width, label='Lateral Downsample Skip', color='#2ecc71')
    ax5.bar(x_indices + width, s2, width, label='Backbone Direct Highway', color='#e67e22')

    ax5.set_xticks(x_indices)
    ax5.set_xticklabels(scales, color='white', fontsize=11, fontweight='bold')
    ax5.set_ylabel("Highway Mixing Weight", color='white')
    ax5.set_ylim(0.0, 1.0)
    ax5.tick_params(colors='white')
    ax5.set_title("Scale-AttnRes Dynamic Flow Balance", color='white', fontsize=14, fontweight='bold')
    ax5.legend(facecolor='#2c3e50', edgecolor='none', labelcolor='white')

    # Panel 6: Counterfactual Attribution
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor('#1e1e1e')
    comp_names = ['Full TDE Model', 'Without Stage 1 TTT', 'Without Engram Memory']
    box_counts = [len(ablations.get('Full', [])), len(ablations.get('No_TTT', [])), len(ablations.get('No_Engram', []))]
    mean_confs = [
        ablations['Full'][:, 4].mean().item() if len(ablations['Full']) > 0 else 0.0,
        ablations['No_TTT'][:, 4].mean().item() if len(ablations['No_TTT']) > 0 else 0.0,
        ablations['No_Engram'][:, 4].mean().item() if len(ablations['No_Engram']) > 0 else 0.0,
    ]

    bars = ax6.barh(comp_names, box_counts, color=['#2ecc71', '#e74c3c', '#9b59b6'], alpha=0.85, height=0.5)
    ax6.set_xlabel("Objects Detected", color='white')
    ax6.tick_params(colors='white')
    ax6.set_title(f"Component Attribution [{domain_tag.upper()}]", color='white', fontsize=14, fontweight='bold')

    for bar, conf in zip(bars, mean_confs):
        w = bar.get_width()
        ax6.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"Conf: {conf:.2f}",
                 va='center', color='white', fontweight='bold', fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
    plt.close()
    print(f"\n{Style.GREEN}✔ Visual Diagnostic Dashboard saved to: {Style.BOLD}{save_path}{Style.RESET}")


def run_live_probe(exp, model, img_path, conf_thresh, nms_thresh, domain_tag, save_dir, device="cuda", fp16=True):
    print(f"\n{Style.BOLD}{Style.WHITE}================================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.WHITE}   TDE-YOLOX v3.2 ADVERSE FORENSIC X-RAY: [{domain_tag.upper()}]                 {Style.RESET}")
    print(f"{Style.BOLD}{Style.WHITE}================================================================================{Style.RESET}")
    print(f"  • Domain:       {Style.BOLD}{Style.YELLOW}{domain_tag.upper()}{Style.RESET}")
    print(f"  • Image Path:   {Style.BOLD}{img_path}{Style.RESET}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"{Style.RED}Error: Failed to load image at {img_path}{Style.RESET}")
        return

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mean_r, mean_g, mean_b = img_rgb[:, :, 0].mean(), img_rgb[:, :, 1].mean(), img_rgb[:, :, 2].mean()
    haze_intensity = (mean_r + mean_g + mean_b) / (3.0 * 255.0)

    preproc = ValTransform(legacy=False)
    img_tensor, _ = preproc(img_bgr, None, exp.test_size)
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device)

    use_fp16 = fp16 and (device.type == "cuda")
    if use_fp16:
        model.half()
        img_tensor = img_tensor.half()
    else:
        model.float()
        img_tensor = img_tensor.float()

    xray = TDEInferenceXRay(model)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "set_meta_training_state"):
        m.set_meta_training_state(exp.max_epoch, exp.max_epoch)

    # Warmup pass
    with torch.no_grad():
        _ = model(img_tensor)
    xray.clear()

    torch.cuda.synchronize()
    start_event.record()

    with torch.no_grad():
        outputs = model(img_tensor)

    end_event.record()
    torch.cuda.synchronize()
    total_fwd_ms = start_event.elapsed_time(end_event)

    nms_start = time.time()
    detections = postprocess(outputs, exp.num_classes, conf_thresh, nms_thresh)[0]
    nms_ms = (time.time() - nms_start) * 1000.0

    ratio = min(exp.test_size[0] / orig_h, exp.test_size[1] / orig_w)
    if detections is not None and len(detections) > 0:
        detections[:, :4] /= ratio
        boxes_np = detections.cpu().numpy()
    else:
        boxes_np = np.zeros((0, 7))

    print(f"\n  Running Component Counterfactual Attribution Analysis...")
    ablations = run_counterfactual_ablation(model, img_tensor, exp)

    # Output detailed report
    print(f"\n{Style.BOLD}{Style.CYAN}--- [1] ATMOSPHERIC & PHOTON SENSING AUDIT ---{Style.RESET}")
    print(f"  • Resolution:            {orig_w}x{orig_h} -> Letterbox {exp.test_size[0]}x{exp.test_size[1]}")
    print(f"  • Koschmieder Haze Index: {Style.BOLD}{haze_intensity:.3f}{Style.RESET} (1.0 = Whiteout / 0.0 = Black)")
    print(f"  • Dynamic Range (RGB):   R:{mean_r:.1f}, G:{mean_g:.1f}, B:{mean_b:.1f}")

    print(f"\n{Style.BOLD}{Style.MAGENTA}--- [2] STAGE 1 ADAPTIVE STEM (TTT-DARK2) AUDIT ---{Style.RESET}")
    print(f"  • TTT Triggered:         {Style.GREEN}YES (Active){Style.RESET}" if xray.taps.get("ttt_executed", False) else f"{Style.RED}NO{Style.RESET}")
    print(f"  • Inner Loop Loss:       {Style.BOLD}{xray.taps.get('ttt_inner_loss', 0.0):.6f}{Style.RESET}")
    print(f"  • Active Mask Ratio:     {xray.taps.get('ttt_mask_ratio', 0.0)*100.0:.2f}% tokens adapted")

    print(f"\n{Style.BOLD}{Style.BLUE}--- [3] BQSA MULTI-SCALE ATTENTION NECK AUDIT ---{Style.RESET}")
    h_s = xray.taps.get("p5_entropy_raw", 0.0)
    tau_adapt = xray.taps.get("tau_adaptive", 0.0)
    print(f"  • P5 Optical Entropy:    {Style.BOLD}{h_s:.4f}{Style.RESET} nats (Max: {math.log(400.0):.4f})")
    print(f"  • Routing Temperature:   tau = {Style.BOLD}{tau_adapt:.4f}{Style.RESET}")

    topk = xray.taps.get("p4_topk_indices", None)
    if topk is not None:
        print(f"  • P4 Top-25 Sparse Blocks: {topk[0].numpy().tolist()[:8]}... (Selected out of 100)")

    print(f"\n{Style.BOLD}{Style.YELLOW}--- [4] SCALE-ATTNRES GRADIENT HIGHWAY AUDIT ---{Style.RESET}")
    for name, label in [("attnres_n3", "N3 Stride 8"), ("attnres_n4", "N4 Stride 16"), ("attnres_n5", "N5 Stride 32")]:
        weights = xray.taps.get(f"{name}_weights", None)
        if weights is not None:
            w_str = ", ".join([f"{v:.3f}" for v in weights])
            print(f"  • {label:<15} [TopDown BQSA, Lateral Skip, Backbone Direct] -> [{w_str}]")

    print(f"\n{Style.BOLD}{Style.GREEN}--- [5] ENGRAM IDENTITY RESTORATION (HEAD) AUDIT ---{Style.RESET}")
    for scale_idx, label in enumerate(["P3 (Small)", "P4 (Medium)", "P5 (Large)"]):
        mean_g = xray.taps.get(f"engram_scale_{scale_idx}_mean_gate", 0.0)
        active_px = xray.taps.get(f"engram_scale_{scale_idx}_active_pixels", 0.0)
        print(f"  • {label:<12} -> Mean Injection Gate: {Style.BOLD}{mean_g:.4f}{Style.RESET} | Active Foregrounds (g > 0.10): {active_px:.2f}%")

    print(f"\n{Style.BOLD}{Style.RED}--- [6] LIVE COMPONENT ATTRIBUTION (WHICH CONTRIBUTES MOST?) ---{Style.RESET}")
    n_full = len(ablations['Full'])
    n_no_ttt = len(ablations['No_TTT'])
    n_no_eng = len(ablations['No_Engram'])
    
    print(f"  • Complete TDE-YOLOX Detections:       {Style.BOLD}{Style.GREEN}{n_full} objects{Style.RESET}")
    print(f"  • Without Stage 1 TTT Adaptation:      {n_no_ttt} objects ({n_full - n_no_ttt:+d} boxes)")
    print(f"  • Without Engram Associative Memory:   {n_no_eng} objects ({n_full - n_no_eng:+d} boxes)")

    if n_full > n_no_ttt and n_full > n_no_eng:
        print(f"  ► {Style.BOLD}{Style.CYAN}Bimodal Synergy: Both TTT Adaptation and Engram Memory rescued detections.{Style.RESET}")
    elif n_full > n_no_ttt:
        print(f"  ► {Style.BOLD}{Style.GREEN}Primary Driver: Stage 1 TTT resolved atmospheric feature distortion.{Style.RESET}")
    elif n_full > n_no_eng:
        print(f"  ► {Style.BOLD}{Style.MAGENTA}Primary Driver: Engram Memory Bank rescued identity classification.{Style.RESET}")
    else:
        print(f"  ► {Style.BOLD}{Style.BLUE}Balanced Equilibrium: Features intact without severe degradation.{Style.RESET}")

    print(f"\n{Style.BOLD}{Style.WHITE}--- [7] LATENCY & REAL-TIME PROFILING ---{Style.RESET}")
    print(f"  • Forward Pass Latency:  {total_fwd_ms:.2f} ms")
    print(f"  • NMS Latency:           {nms_ms:.2f} ms")
    print(f"  • Total Frame Latency:   {Style.BOLD}{Style.GREEN}{total_fwd_ms + nms_ms:.2f} ms{Style.RESET} ({1000.0 / (total_fwd_ms + nms_ms):.1f} FPS)")

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    dash_save_path = os.path.join(save_dir, f"{domain_tag}_{img_name}_XRAY_DASHBOARD.png")
    render_diagnostic_dashboard(img_rgb, boxes_np, xray.taps, ablations, domain_tag, dash_save_path)

    # Machine-parseable telemetry line for master summary aggregation
    g_p3 = xray.taps.get("engram_scale_0_mean_gate", 0.0)
    inner_l = xray.taps.get("ttt_inner_loss", 0.0)
    mask_r = xray.taps.get("ttt_mask_ratio", 0.0) * 100.0
    print(f"__TELEMETRY__|{domain_tag}|{haze_intensity:.3f}|{h_s:.4f}|{mask_r:.1f}|{inner_l:.6f}|{g_p3:.4f}|{n_full}|{n_no_ttt}|{n_no_eng}")


def main():
    parser = argparse.ArgumentParser("TDE-YOLOX v3.2 Cross-Domain Adverse Weather Diagnostic Probe")
    parser.add_argument("-f", "--exp_file", type=str, required=True, help="Path to experiment file")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to target input image")
    parser.add_argument("--domain", type=str, default="adverse", help="Domain category (fog, snow, rain, night, clean)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.65, help="NMS IoU threshold")
    parser.add_argument("--tsize", type=int, default=640, help="Test image size")
    parser.add_argument("--save_dir", type=str, default="./probe_results", help="Directory for output")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--fp16", dest="fp16", default=True, action="store_true", help="Run inference in FP16")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Run inference in FP32")
    parser.add_argument("--log_file", type=str, default=None, help="Custom output .log file path")
    args = parser.parse_args()

    default_log_file = args.log_file if args.log_file else os.path.join(args.save_dir, "probe_inference.log")
    logger_instance = DualLogger(default_log_file)
    sys.stdout = logger_instance

    try:
        exp = get_exp(args.exp_file, None)
        exp.test_conf = args.conf
        exp.nmsthre = args.nms
        exp.test_size = (args.tsize, args.tsize)

        model = exp.get_model()
        device = torch.device(args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
        model.to(device)
        model.eval()

        print(f"Loading checkpoint from: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        if isinstance(ckpt, dict) and "ema" in ckpt and ckpt["ema"] is not None:
            ckpt_state = ckpt["ema"]
            print("  • Loaded ModelEMA shadow weights for inference probe.")
        elif isinstance(ckpt, dict) and "model" in ckpt:
            ckpt_state = ckpt["model"]
            print("  • Loaded standard model weights for inference probe.")
        else:
            ckpt_state = ckpt

        model = load_ckpt(model, ckpt_state)

        run_live_probe(
            exp=exp,
            model=model,
            img_path=args.image,
            conf_thresh=args.conf,
            nms_thresh=args.nms,
            domain_tag=args.domain,
            save_dir=args.save_dir,
            device=device,
            fp16=args.fp16
        )
    finally:
        sys.stdout = logger_instance.terminal
        logger_instance.close()
        print(f"\n{Style.GREEN}✔ Complete session output recorded to: {Style.BOLD}{default_log_file}{Style.RESET}")


if __name__ == "__main__":
    main()