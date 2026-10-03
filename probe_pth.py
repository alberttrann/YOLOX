#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Forensic PyTorch Checkpoint (.pth) Inspector & Deep Model State Probe

import os
import sys
import math
import argparse
from collections import OrderedDict, defaultdict
import torch
import torch.nn.functional as F
import sys
import os
import re

# === WINDOWS UTF-8 & ANSI SHIELD ===
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

class AnsiStrippingFileLogger:
    """Tees output to terminal with colors, but strips ANSI codes for file output."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8", errors="replace")
        self.ansi_re = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

    def write(self, message):
        self.terminal.write(message)
        # Strip color codes before writing to disk
        clean_msg = self.ansi_re.sub('', message)
        self.log_file.write(clean_msg)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        if self.log_file and not self.log_file.closed:
            self.log_file.close()
# ANSI Terminal Styler
class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size) < 1024.0:
            return f"{size:3.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def format_num(n):
    if abs(n) >= 1e9:
        return f"{n/1e9:.2f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def inspect_tensor_stats(t):
    """Computes full numerical statistics with zero FP16 accumulator overflow."""
    if not torch.is_tensor(t):
        return None

    numel = t.numel()
    if numel == 0:
        return {"numel": 0, "shape": list(t.shape), "dtype": str(t.dtype), "nans": 0, "infs": 0}

    # Execute health checks strictly in FP32/FP64
    t_f = t.detach().float()
    nans = int(torch.isnan(t_f).sum().item())
    infs = int(torch.isinf(t_f).sum().item())
    
    if nans == numel:
        return {
            "numel": numel, "shape": list(t.shape), "dtype": str(t.dtype),
            "nans": nans, "infs": infs, "min": float('nan'), "max": float('nan'),
            "mean": float('nan'), "std": float('nan'), "norm": float('nan'), "zeros": 0.0
        }

    valid_mask = torch.isfinite(t_f)
    if valid_mask.any():
        t_valid = t_f[valid_mask]
        t_min = t_valid.min().item()
        t_max = t_valid.max().item()
        t_mean = t_valid.mean().item()
        t_std = t_valid.std(unbiased=False).item() if t_valid.numel() > 1 else 0.0
        t_norm = torch.norm(t_valid, p=2).item()
        zero_ratio = (t_valid == 0.0).float().mean().item() * 100.0
    else:
        t_min = t_max = t_mean = t_std = t_norm = zero_ratio = float('nan')

    return {
        "numel": numel,
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "nans": nans,
        "infs": infs,
        "min": t_min,
        "max": t_max,
        "mean": t_mean,
        "std": t_std,
        "norm": t_norm,
        "zeros": zero_ratio
    }


def probe_prototypes(state_dict):
    """Specialized Probe: Hyperspherical S^127 Prototypes Analysis."""
    proto_keys = [k for k in state_dict.keys() if "prototypes" in k]
    if not proto_keys:
        return

    print(f"\n{Style.BOLD}{Style.MAGENTA}================================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.MAGENTA}   [SPECIAL PROBE] HYPERSPHERICAL PROTOTYPES (S^127 MANIFOLD GEOMETRY)           {Style.RESET}")
    print(f"{Style.BOLD}{Style.MAGENTA}================================================================================{Style.RESET}")

    for k in proto_keys:
        P = state_dict[k].float()
        print(f"\n{Style.BOLD}• Tensor:{Style.RESET} {k} | Shape: {list(P.shape)} | Elements: {P.numel():,}")
        
        # Check row-wise L2 norm
        row_norms = torch.norm(P, p=2, dim=-1)
        mean_norm = row_norms.mean().item()
        min_norm = row_norms.min().item()
        max_norm = row_norms.max().item()
        norm_dev = (row_norms - 1.0).abs().max().item()

        status = f"{Style.GREEN}PERFECT (On S^127){Style.RESET}" if norm_dev < 1e-3 else f"{Style.YELLOW}UNNORMALIZED (Dev={norm_dev:.4f}){Style.RESET}"
        print(f"  - Unit Hypersphere Check: {status} | Mean ||p||: {mean_norm:.6f} (Min: {min_norm:.6f}, Max: {max_norm:.6f})")

        # Hyperspherical Cosine Similarity & Angular Separation
        P_norm = F.normalize(P, p=2, dim=-1)
        sim_matrix = torch.matmul(P_norm, P_norm.t())
        
        # Exclude self-similarity diagonal
        eye = torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
        off_diag = sim_matrix[~eye]

        mean_sim = off_diag.mean().item()
        max_sim = off_diag.max().item()
        min_sim = off_diag.min().item()
        
        # Mean angular distance in degrees
        clamped_sim = torch.clamp(off_diag, -1.0, 1.0)
        mean_angle_deg = torch.acos(clamped_sim).mean().item() * (180.0 / math.pi)

        print(f"  - Inter-Prototype Cosine: Mean={mean_sim:+.4f} | Max={max_sim:+.4f} | Min={min_sim:+.4f}")
        print(f"  - Mean Angular Distance:  {Style.CYAN}{mean_angle_deg:.2f}°{Style.RESET} (Orthogonal = 90.00°)")


def probe_ttt_engine(state_dict):
    """Specialized Probe: TTT-Adaptive Stage 1 Learning Rates."""
    ttt_keys = [k for k in state_dict.keys() if "ttt_lrs" in k]
    if not ttt_keys:
        return

    print(f"\n{Style.BOLD}{Style.CYAN}================================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}   [SPECIAL PROBE] TTT-DARK2 STAGE 1 META-LEARNING STEP SIZES                    {Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}================================================================================{Style.RESET}")

    for k in ttt_keys:
        val = state_dict[k].float()
        effective_lr = F.softplus(val).item()
        raw_val = val.item() if val.numel() == 1 else val.mean().item()
        print(f"  • {k:<45} | Raw: {raw_val:+.4f} | Effective Inner LR: {Style.BOLD}{Style.YELLOW}{effective_lr:.6f}{Style.RESET}")


def probe_ema_drift(model_dict, ema_dict):
    """Specialized Probe: Parameter-by-Parameter Model vs. ModelEMA Drift Analysis."""
    print(f"\n{Style.BOLD}{Style.YELLOW}================================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.YELLOW}   [SPECIAL PROBE] MODEL vs. MODEL-EMA PARAMETER DRIFT ANALYSIS                  {Style.RESET}")
    print(f"{Style.BOLD}{Style.YELLOW}================================================================================{Style.RESET}")

    common_keys = [k for k in model_dict.keys() if k in ema_dict]
    if not common_keys:
        print("  No common parameter keys found between live model and EMA!")
        return

    drifts = []
    for k in common_keys:
        m_t = model_dict[k].float()
        e_t = ema_dict[k].float()
        diff = torch.norm(m_t - e_t, p=2).item()
        norm_m = torch.norm(m_t, p=2).item() + 1e-7
        rel_drift = (diff / norm_m) * 100.0
        drifts.append((k, diff, rel_drift))

    drifts.sort(key=lambda x: -x[2])

    print(f"  Shared Tensors Analyzed: {len(common_keys)}")
    print(f"\n  {Style.BOLD}Top 5 Most Dynamically Drifted Parameters (Live vs. EMA):{Style.RESET}")
    for k, diff, rel in drifts[:5]:
        print(f"    - {k:<55} | L2 Diff: {diff:.4e} | Rel Drift: {Style.RED}{rel:.2f}%{Style.RESET}")

    print(f"\n  {Style.BOLD}Top 5 Most Rigid / Stable Parameters (Live vs. EMA):{Style.RESET}")
    for k, diff, rel in drifts[-5:]:
        print(f"    - {k:<55} | L2 Diff: {diff:.4e} | Rel Drift: {Style.GREEN}{rel:.2f}%{Style.RESET}")


def analyze_state_dict(sd, name="MODEL STATE DICT", verbose=False, filter_pattern=None):
    print(f"\n{Style.BOLD}{Style.BLUE}================================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.BLUE}   {name} SUMMARY                                                      {Style.RESET}")
    print(f"{Style.BOLD}{Style.BLUE}================================================================================{Style.RESET}")

    total_params = 0
    total_bytes = 0
    dtype_counts = defaultdict(int)
    module_groups = defaultdict(lambda: {"count": 0, "params": 0, "bytes": 0})
    nan_tensors = []
    inf_tensors = []
    dead_tensors = []

    rows = []

    for k, v in sd.items():
        if filter_pattern and filter_pattern not in k:
            continue

        stats = inspect_tensor_stats(v)
        if stats is None:
            continue

        numel = stats["numel"]
        total_params += numel
        t_bytes = v.element_size() * numel
        total_bytes += t_bytes
        dtype_counts[stats["dtype"]] += numel

        prefix = k.split('.')[0] if '.' in k else "root"
        if prefix == "backbone" and '.' in k:
            parts = k.split('.')
            prefix = f"{parts[0]}.{parts[1]}"
        elif prefix == "head" and '.' in k:
            parts = k.split('.')
            prefix = f"{parts[0]}.{parts[1]}"

        module_groups[prefix]["count"] += 1
        module_groups[prefix]["params"] += numel
        module_groups[prefix]["bytes"] += t_bytes

        if stats["nans"] > 0:
            nan_tensors.append((k, stats["nans"]))
        if stats["infs"] > 0:
            inf_tensors.append((k, stats["infs"]))
        if stats["zeros"] == 100.0 and numel > 1:
            dead_tensors.append(k)

        rows.append((k, stats))

    print(f"  • Total Tensors:       {Style.BOLD}{len(sd):,}{Style.RESET}")
    print(f"  • Total Parameter Count: {Style.BOLD}{Style.GREEN}{total_params:,} ({format_num(total_params)}){Style.RESET}")
    print(f"  • Tensor Memory in RAM:  {Style.BOLD}{format_bytes(total_bytes)}{Style.RESET}")
    
    dtypes_str = ", ".join([f"{k}: {format_num(v)}" for k, v in dtype_counts.items()])
    print(f"  • Dtype Distribution:  {dtypes_str}")

    # Health Alert
    print(f"\n  {Style.BOLD}Numerical Health Audit:{Style.RESET}")
    if nan_tensors:
        print(f"    {Style.RED}✖ WARNING: Found {len(nan_tensors)} tensors containing NaN values!{Style.RESET}")
        for t_name, cnt in nan_tensors[:5]:
            print(f"       - {t_name} ({cnt} NaNs)")
    else:
        print(f"    {Style.GREEN}✔ Zero NaN values detected across all tensors.{Style.RESET}")

    if inf_tensors:
        print(f"    {Style.RED}✖ WARNING: Found {len(inf_tensors)} tensors containing Inf values!{Style.RESET}")
        for t_name, cnt in inf_tensors[:5]:
            print(f"       - {t_name} ({cnt} Infs)")
    else:
        print(f"    {Style.GREEN}✔ Zero Inf values detected across all tensors.{Style.RESET}")

    if dead_tensors:
        print(f"    {Style.YELLOW}⚠ Notice: Found {len(dead_tensors)} all-zero tensors (dead/uninitialized):{Style.RESET}")
        for t_name in dead_tensors[:5]:
            print(f"       - {t_name}")
    else:
        print(f"    {Style.GREEN}✔ Zero completely dead tensors found.{Style.RESET}")

    # Module Group Hierarchy Table
    print(f"\n  {Style.BOLD}Subsystem Parameter Partitioning:{Style.RESET}")
    print(f"  {'Subsystem Prefix':<35} | {'Tensors':<8} | {'Param Count':<14} | {'Memory':<10}")
    print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*14}-+-{'-'*10}")
    for grp, d in sorted(module_groups.items(), key=lambda x: -x[1]["params"]):
        print(f"  {grp:<35} | {d['count']:<8} | {format_num(d['params']):<14} | {format_bytes(d['bytes']):<10}")

    # Detailed Per-Tensor Printout
    if verbose:
        print(f"\n  {Style.BOLD}Individual Tensor Inspection:{Style.RESET}")
        print(f"  {'Tensor Name':<55} | {'Shape':<18} | {'Min':<9} | {'Mean':<9} | {'Max':<9} | {'L2 Norm':<9} | {'Zeros %':<7}")
        print(f"  {'-'*55}-+-{'-'*18}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}")
        for k, s in rows:
            shape_str = str(s['shape'])
            if len(shape_str) > 18:
                shape_str = shape_str[:15] + "..."
            print(f"  {k:<55} | {shape_str:<18} | {s['min']:+8.2e} | {s['mean']:+8.2e} | {s['max']:+8.2e} | {s['norm']:8.2e} | {s['zeros']:5.1f}%")


def inspect_optimizer(opt_state):
    """Exhaustive Optimizer State Inspection (SGD, Adam, Muon, MuSGD)."""
    print(f"\n{Style.BOLD}{Style.CYAN}================================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}   OPTIMIZER STATE AUDIT                                                        {Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}================================================================================{Style.RESET}")

    if not isinstance(opt_state, dict):
        print(f"  Optimizer state is of type: {type(opt_state)}")
        return

    param_groups = opt_state.get("param_groups", [])
    print(f"  • Registered Parameter Groups: {Style.BOLD}{len(param_groups)}{Style.RESET}")
    
    for idx, grp in enumerate(param_groups):
        p_count = len(grp.get("params", []))
        lr = grp.get("lr", "N/A")
        wd = grp.get("weight_decay", "N/A")
        mom = grp.get("momentum", "N/A")
        use_muon = grp.get("use_muon", False)
        lr_factor = grp.get("lr_factor", "N/A")
        
        opt_type = f"{Style.MAGENTA}Muon (Newton-Schulz){Style.RESET}" if use_muon else f"{Style.BLUE}Standard SGD/Momentum{Style.RESET}"
        print(f"\n  {Style.BOLD}Group {idx}:{Style.RESET} Type: {opt_type} | Params Tracked: {p_count}")
        print(f"    - Base LR: {Style.BOLD}{lr}{Style.RESET} | LR Factor: {lr_factor}")
        print(f"    - Momentum: {mom} | Weight Decay: {wd} | Nesterov: {grp.get('nesterov', False)}")

    state = opt_state.get("state", {})
    total_momentum_bytes = 0
    buffer_types = defaultdict(int)
    
    for pid, s in state.items():
        for b_name, b_tensor in s.items():
            if torch.is_tensor(b_tensor):
                total_momentum_bytes += b_tensor.element_size() * b_tensor.numel()
                buffer_types[b_name] += 1

    print(f"\n  • Optimizer Internal State Tensors:")
    print(f"    - Unique Tracked Parameters with State: {len(state):,}")
    print(f"    - Total Momentum/Buffer Memory:        {Style.BOLD}{format_bytes(total_momentum_bytes)}{Style.RESET}")
    for b_name, cnt in buffer_types.items():
        print(f"      * {b_name}: {cnt} tensors")


def main():
    parser = argparse.ArgumentParser(description="Deep Forensic PyTorch Checkpoint (.pth) Inspector")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print granular line-by-line tensor stats")
    parser.add_argument("--filter", type=str, default=None, help="Filter tensor names by substring regex")
    parser.add_argument("--log_file", type=str, default=None, help="Save clean text output to file")
    args = parser.parse_args()

    if args.log_file:
        sys.stdout = AnsiStrippingFileLogger(args.log_file)

    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.exists(ckpt_path):
        print(f"{Style.RED}Error: File does not exist at {ckpt_path}{Style.RESET}")
        sys.exit(1)

    file_size = os.path.getsize(ckpt_path)

    print(f"{Style.BOLD}{Style.WHITE}╔══════════════════════════════════════════════════════════════════════════════╗{Style.RESET}")
    print(f"{Style.BOLD}{Style.WHITE}║               TDE-YOLOX FORENSIC CHECKPOINT PROBE TOOL                       ║{Style.RESET}")
    print(f"{Style.BOLD}{Style.WHITE}╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET}")
    print(f"  • File Path:   {Style.BOLD}{ckpt_path}{Style.RESET}")
    print(f"  • Disk Size:   {Style.BOLD}{format_bytes(file_size)}{Style.RESET}")

    # Load safely to CPU
    print("\n  Loading checkpoint to CPU memory...")
    try:
        # PyTorch 2.4+ safe loading override
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception as e2:
            print(f"{Style.RED}Failed to load checkpoint: {e2}{Style.RESET}")
            sys.exit(1)

    print(f"  Checkpoint Object Type: {Style.BOLD}{type(ckpt)}{Style.RESET}")

    # Case 1: Complex Dictionary (YOLOX Full Checkpoint)
    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        print(f"\n  {Style.BOLD}Top-Level Checkpoint Keys Found:{Style.RESET}")
        for k in keys:
            v_type = type(ckpt[k])
            if isinstance(ckpt[k], dict):
                content_info = f"dict with {len(ckpt[k])} keys"
            elif torch.is_tensor(ckpt[k]):
                content_info = f"Tensor of shape {list(ckpt[k].shape)}"
            else:
                content_info = f"{ckpt[k]}"
            print(f"    • {Style.CYAN}{k:<20}{Style.RESET} : {content_info}")

        # Metadata Summary
        meta_items = [
            ("start_epoch", ckpt.get("start_epoch")),
            ("max_epoch", ckpt.get("max_epoch")),
            ("best_ap", ckpt.get("best_ap")),
            ("curr_ap", ckpt.get("curr_ap")),
            ("curr_iter", ckpt.get("curr_iter")),
        ]
        active_meta = [(k, v) for k, v in meta_items if v is not None]
        if active_meta:
            print(f"\n  {Style.BOLD}Serialized Training Metadata:{Style.RESET}")
            for k, v in active_meta:
                if isinstance(v, float):
                    print(f"    - {k:<15} : {v:.4f}")
                else:
                    print(f"    - {k:<15} : {v}")

        # 1. Probe Model State Dict
        model_sd = ckpt.get("model", None)
        if isinstance(model_sd, dict):
            analyze_state_dict(model_sd, name="LIVE MODEL STATE DICT (ckpt['model'])", verbose=args.verbose, filter_pattern=args.filter)
            probe_prototypes(model_sd)
            probe_ttt_engine(model_sd)

        # 2. Probe EMA State Dict
        ema_sd = ckpt.get("ema", None)
        if isinstance(ema_sd, dict):
            analyze_state_dict(ema_sd, name="MODEL-EMA STATE DICT (ckpt['ema'])", verbose=args.verbose, filter_pattern=args.filter)
            probe_prototypes(ema_sd)
            
            # Compare Live Model vs EMA drift
            if isinstance(model_sd, dict):
                probe_ema_drift(model_sd, ema_sd)

        # 3. Probe Optimizer
        opt_state = ckpt.get("optimizer", None)
        if opt_state:
            inspect_optimizer(opt_state)

        # 4. Probe Scaler (AMP)
        scaler_state = ckpt.get("scaler", None)
        if scaler_state and isinstance(scaler_state, dict):
            print(f"\n{Style.BOLD}{Style.GREEN}================================================================================{Style.RESET}")
            print(f"{Style.BOLD}{Style.GREEN}   AMP GRAD-SCALER STATE                                                        {Style.RESET}")
            print(f"{Style.BOLD}{Style.GREEN}================================================================================{Style.RESET}")
            for sk, sv in scaler_state.items():
                print(f"  • {sk:<25}: {sv}")

    # Case 2: Raw State Dict (e.g. weights only, yolox_s.pth)
    elif isinstance(ckpt, (dict, OrderedDict)):
        analyze_state_dict(ckpt, name="RAW STATE DICT", verbose=args.verbose, filter_pattern=args.filter)
        probe_prototypes(ckpt)
        probe_ttt_engine(ckpt)

    print(f"\n{Style.BOLD}{Style.WHITE}================================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.GREEN}   FORENSIC PROBE COMPLETE.                                                     {Style.RESET}")
    print(f"{Style.BOLD}{Style.WHITE}================================================================================{Style.RESET}\n")


if __name__ == "__main__":
    main()