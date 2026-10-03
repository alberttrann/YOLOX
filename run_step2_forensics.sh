#!/usr/bin/env bash
# ==============================================================================
# TDE-YOLOX v3.2 Step 2: Longitudinal Static Parameter & Optimizer Forensics
# Automated 5-Milestone Parameter Drift & Manifold Distortion Audit
# ==============================================================================

set -eo pipefail

export PYTHONIOENCODING="utf-8"
export PYTHONUTF8=1
export LANG="en_US.UTF-8"

# ------------------------------------------------------------------------------
# 0. ROBUST PYTHON VIRTUALENV AUTO-DETECTION
# ------------------------------------------------------------------------------
if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/Scripts/python.exe" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/Scripts/python.exe"
elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
elif [ -f ".venv/Scripts/python" ]; then
    PYTHON_BIN=".venv/Scripts/python"
elif [ -f "venv/Scripts/python.exe" ]; then
    PYTHON_BIN="venv/Scripts/python.exe"
elif [ -f "../.venv/Scripts/python.exe" ]; then
    PYTHON_BIN="../.venv/Scripts/python.exe"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

# Verify PyTorch availability in detected python binary
if ! "$PYTHON_BIN" -c "import torch" &>/dev/null; then
    echo -e "\033[31mError: PyTorch not found using Python at: ${PYTHON_BIN}\033[0m"
    echo "Please activate your virtual environment or set PYTHON_BIN manually."
    exit 1
fi

CKPT_DIR="YOLOX_outputs/tde_yolox_s"
OUT_BASE="./probe_results/step2_static_forensics"

mkdir -p "${OUT_BASE}"

# Terminal Styler
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
MAGENTA="\033[35m"
CYAN="\033[36m"
RED="\033[31m"
RESET="\033[0m"

print_header() {
    echo -e "\n${BOLD}${BLUE}================================================================================${RESET}"
    echo -e "${BOLD}${BLUE}  $1${RESET}"
    echo -e "${BOLD}${BLUE}================================================================================${RESET}\n"
}

print_step() {
    echo -e "${BOLD}${CYAN}==> [PROBING]:${RESET} ${YELLOW}$1${RESET}"
}

print_header "TDE-YOLOX STEP 2: LONGITUDINAL STATIC FORENSICS INITIATION"
echo -e "Using Python Binary: ${BOLD}${GREEN}${PYTHON_BIN}${RESET}"

# ------------------------------------------------------------------------------
# 1. MILESTONE CHECKPOINT AUTO-RESOLVER
# ------------------------------------------------------------------------------
resolve_milestone() {
    local primary="$1"
    local fallback="$2"
    if [ -f "${CKPT_DIR}/${primary}" ]; then
        echo "${CKPT_DIR}/${primary}"
    elif [ -f "${CKPT_DIR}/${fallback}" ]; then
        echo "${CKPT_DIR}/${fallback}"
    else
        echo ""
    fi
}

M1_CKPT=$(resolve_milestone "epoch_10_ckpt.pth" "epoch_10_ckpt.pth")
M2_CKPT=$(resolve_milestone "epoch_24_ckpt.pth" "best_ckpt.pth")
M3_CKPT=$(resolve_milestone "epoch_40_ckpt.pth" "epoch_38_ckpt.pth")
M4_CKPT=$(resolve_milestone "epoch_64_ckpt.pth" "epoch_65_ckpt.pth")
M5_CKPT=$(resolve_milestone "epoch_78_ckpt.pth" "latest_ckpt.pth")

# Export variables for downstream Python synthesizer
export M1_CKPT M2_CKPT M3_CKPT M4_CKPT M5_CKPT

TAGS=("M1_Warmup_E10" "M2_Peak_E24" "M3_Inflection_E40" "M4_SagFloor_E64" "M5_Terminal_E78")
PATHS=("$M1_CKPT" "$M2_CKPT" "$M3_CKPT" "$M4_CKPT" "$M5_CKPT")

echo -e "\n${BOLD}Target Milestones Resolved:${RESET}"
for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"
    path="${PATHS[$i]}"
    if [ -n "$path" ] && [ -f "$path" ]; then
        printf "  • ${CYAN}%-22s${RESET} -> ${GREEN}%s${RESET}\n" "$tag" "$path"
    else
        printf "  • ${CYAN}%-22s${RESET} -> ${RED}NOT FOUND in %s${RESET}\n" "$tag" "$CKPT_DIR"
        echo -e "${RED}Error: Required milestone checkpoint missing. Halting.${RESET}"
        exit 1
    fi
done

# ------------------------------------------------------------------------------
# 2. RUN FULL GRANULAR PROBES ON ALL 5 MILESTONES
# ------------------------------------------------------------------------------
for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"
    ckpt="${PATHS[$i]}"
    milestone_out="${OUT_BASE}/${tag}"
    mkdir -p "${milestone_out}"

    print_header "Executing Forensic Probes on: [${tag}] (${ckpt})"

    # 1. Full Model & Optimizer Audit
    print_step "Full System Overview & Optimizer State"
    ${PYTHON_BIN} probe_pth.py -c "${ckpt}" --log_file "${milestone_out}/00_full_audit.log"

    # 2. Memory Bank & Prototypes (Hyperspherical Geometry)
    print_step "Engram Memory Bank & Prototypes (S^127)"
    ${PYTHON_BIN} probe_pth.py -c "${ckpt}" -v --filter "memory_banks" --log_file "${milestone_out}/01_memory_banks.log"

    # 3. Stage 1 TTT-Dark2 & Meta-LRs
    print_step "Stage 1 TTT-Dark2 GroupNorm & Meta-LRs"
    ${PYTHON_BIN} probe_pth.py -c "${ckpt}" -v --filter "dark2" --log_file "${milestone_out}/02_dark2_ttt.log"

    # 4. BQSA Attention Neck
    print_step "BQSA Attention Neck (P4/P3 Blocks & P5 RoPE)"
    ${PYTHON_BIN} probe_pth.py -c "${ckpt}" -v --filter "bqsa" --log_file "${milestone_out}/03_bqsa_neck.log"

    # 5. Scale-AttnRes Highway
    print_step "Scale-AttnRes Highway Flow Weights"
    ${PYTHON_BIN} probe_pth.py -c "${ckpt}" -v --filter "attnres" --log_file "${milestone_out}/04_scale_attnres.log"

    # 6. Decoupled Prediction Heads
    print_step "Decoupled Classification & Regression Heads"
    ${PYTHON_BIN} probe_pth.py -c "${ckpt}" -v --filter "preds" --log_file "${milestone_out}/05_prediction_heads.log"
done

# ------------------------------------------------------------------------------
# 3. AUTOMATED LONGITUDINAL SYNTHESIS: DIRECT TENSOR CROSS-ANALYSIS
# ------------------------------------------------------------------------------
print_header "COMPILING LONGITUDINAL TRAJECTORY MATRIX (TENSOR DIFF ANALYSIS)"

SYNTHESIS_REPORT="${OUT_BASE}/LONGITUDINAL_TRAJECTORY_MATRIX.txt"

${PYTHON_BIN} - << 'EOF' > "${SYNTHESIS_REPORT}"
import os, sys, math, torch
import torch.nn.functional as F

ckpts = {
    "E10 (Warmup)": os.environ.get("M1_CKPT", "YOLOX_outputs/tde_yolox_s/epoch_10_ckpt.pth"),
    "E24 (Peak)":   os.environ.get("M2_CKPT", "YOLOX_outputs/tde_yolox_s/epoch_24_ckpt.pth"),
    "E40 (SagOn)":  os.environ.get("M3_CKPT", "YOLOX_outputs/tde_yolox_s/epoch_40_ckpt.pth"),
    "E64 (Floor)":  os.environ.get("M4_CKPT", "YOLOX_outputs/tde_yolox_s/epoch_64_ckpt.pth"),
    "E78 (Final)":  os.environ.get("M5_CKPT", "YOLOX_outputs/tde_yolox_s/epoch_78_ckpt.pth"),
}

data = {}
for tag, path in ckpts.items():
    if not os.path.exists(path):
        continue
    c = torch.load(path, map_location="cpu", weights_only=False)
    data[tag] = c

keys = list(data.keys())
print("====================================================================================================================")
print("                      TDE-YOLOX v3.2 LONGITUDINAL PARAMETER EVOLUTION MATRIX (5 MILESTONES)")
print("====================================================================================================================")
print(f"{'Monitored Invariant / Parameter':<42} | " + " | ".join([f"{k:^12}" for k in keys]) + " | Delta (Peak->End)")
print("-" * 116)

def get_tensor(sd, target_suffix):
    if not sd:
        return None
    if target_suffix in sd:
        return sd[target_suffix]
    mod_k = f"module.{target_suffix}"
    if mod_k in sd:
        return sd[mod_k]
    for k, v in sd.items():
        if k.endswith(target_suffix):
            return v
    return None

# 1. Prototype Geometry (Bank 0 - P3)
def get_proto_metrics(sd):
    t = get_tensor(sd, "head.memory_banks.0.prototypes")
    if t is None: 
        return (0.0, 0.0, 0.0)
    P = t.float()
    norms = torch.norm(P, p=2, dim=-1)
    mean_norm = norms.mean().item()
    P_norm = F.normalize(P, p=2, dim=-1)
    sim = torch.matmul(P_norm, P_norm.t())
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    off_diag = sim[~eye]
    mean_angle = torch.acos(torch.clamp(off_diag, -1.0, 1.0)).mean().item() * (180.0 / math.pi)
    max_sim = off_diag.max().item()
    return (mean_norm, mean_angle, max_sim)

# Live Prototypes
vals_norm = [f"{get_proto_metrics(data[k].get('model', {}))[0]:.4f}" for k in keys]
p_delta = float(vals_norm[-1]) - float(vals_norm[1])
print(f"{'P3 Proto Mean Radius ||p|| (Target: 1.0000)':<42} | " + " | ".join([f"{v:^12}" for v in vals_norm]) + f" | {p_delta:+.4f}")

vals_angle = [f"{get_proto_metrics(data[k].get('model', {}))[1]:.2f}°" for k in keys]
p_ang_delta = float(vals_angle[-1].replace('°','')) - float(vals_angle[1].replace('°',''))
print(f"{'P3 Proto Separation Angle (Target: 90.00°)':<42} | " + " | ".join([f"{v:^12}" for v in vals_angle]) + f" | {p_ang_delta:+.2f}°")

vals_sim = [f"{get_proto_metrics(data[k].get('model', {}))[2]:.4f}" for k in keys]
p_sim_delta = float(vals_sim[-1]) - float(vals_sim[1])
print(f"{'P3 Proto Peak Intra-Mode Cosine':<42} | " + " | ".join([f"{v:^12}" for v in vals_sim]) + f" | {p_sim_delta:+.4f}")

# ModelEMA Prototypes
vals_ema_norm = []
for k in keys:
    ema_sd = data[k].get('ema', {})
    t = get_tensor(ema_sd, "head.memory_banks.0.prototypes")
    if t is not None:
        v = torch.norm(t.float(), p=2, dim=-1).mean().item()
        vals_ema_norm.append(f"{v:.4f}")
    else:
        vals_ema_norm.append("N/A")
ema_last = float(vals_ema_norm[-1]) if vals_ema_norm[-1] != "N/A" else 1.0
print(f"{'P3 ModelEMA Proto Radius (Target: 1.0000)':<42} | " + " | ".join([f"{v:^12}" for v in vals_ema_norm]) + f" | Dev: {ema_last - 1.0:+.4f}")

print("-" * 116)

# 2. Stage 1 GroupNorm Affine Weight Gain Drift (Dark2)
def get_gn_gain(sd):
    t = get_tensor(sd, "dark2.backbone_stage.0.gn.weight")
    return t.float().mean().item() if t is not None else 1.0

vals_gn = [f"{get_gn_gain(data[k].get('model', {})):.4f}" for k in keys]
gn_delta = float(vals_gn[-1]) - float(vals_gn[1])
print(f"{'Dark2 GN Entry Gamma Gain (Init: 1.0000)':<42} | " + " | ".join([f"{v:^12}" for v in vals_gn]) + f" | {gn_delta:+.4f}")

# 3. TTT Meta-Learning Step Size
def get_ttt_lr(sd):
    t = get_tensor(sd, "dark2.ttt_lrs.0_gn_weight")
    return F.softplus(t.float()).item() if t is not None else 0.05

vals_ttt = [f"{get_ttt_lr(data[k].get('model', {})):.4f}" for k in keys]
ttt_delta = float(vals_ttt[-1]) - float(vals_ttt[1])
print(f"{'Dark2 TTT Inner LR (Init: 0.0500)':<42} | " + " | ".join([f"{v:^12}" for v in vals_ttt]) + f" | {ttt_delta:+.4f}")

print("-" * 116)

# 4. Scale-AttnRes Query Weight Norms
for res_idx, r_name in [(3, "N3 Stride 8"), (4, "N4 Stride 16"), (5, "N5 Stride 32")]:
    target_k = f"attnres_n{res_idx}.w"
    def get_attnres_norm(sd):
        t = get_tensor(sd, target_k)
        return torch.norm(t.float(), p=2).item() if t is not None else 0.0
    vals_r = [f"{get_attnres_norm(data[k].get('model', {})):.4f}" for k in keys]
    r_delta = float(vals_r[-1]) - float(vals_r[1])
    print(f"{'AttnRes ' + r_name + ' ||w||_2':<42} | " + " | ".join([f"{v:^12}" for v in vals_r]) + f" | {r_delta:+.4f}")

print("-" * 116)

# 5. Engram Head Memory Parameters (omega distance penalty & w_hf)
for b_idx in [0, 2]:
    b_label = "P3 (Small)" if b_idx == 0 else "P5 (Large)"
    om_k = f"head.memory_banks.{b_idx}.omega"
    def get_om(sd):
        t = get_tensor(sd, om_k)
        return t.item() if t is not None else 0.0
    vals_om = [f"{get_om(data[k].get('model', {})):.4f}" for k in keys]
    om_delta = float(vals_om[-1]) - float(vals_om[1])
    print(f"{'Engram ' + b_label + ' omega (Dist Penalty)':<42} | " + " | ".join([f"{v:^12}" for v in vals_om]) + f" | {om_delta:+.4f}")

    hf_k = f"head.memory_banks.{b_idx}.w_hf"
    def get_hf(sd):
        t = get_tensor(sd, hf_k)
        return t.item() if t is not None else 0.0
    vals_hf = [f"{get_hf(data[k].get('model', {})):.4f}" for k in keys]
    hf_delta = float(vals_hf[-1]) - float(vals_hf[1])
    print(f"{'Engram ' + b_label + ' w_hf (High-Freq Weight)':<42} | " + " | ".join([f"{v:^12}" for v in vals_hf]) + f" | {hf_delta:+.4f}")

print("-" * 116)

# 6. Prediction Head Output Biases
for p_idx, p_name in [(0, "P3 Stride 8"), (1, "P4 Stride 16"), (2, "P5 Stride 32")]:
    obj_k = f"head.obj_preds.{p_idx}.bias"
    def get_obj_b(sd):
        t = get_tensor(sd, obj_k)
        return t.item() if t is not None else 0.0
    vals_obj = [f"{get_obj_b(data[k].get('model', {})):.4f}" for k in keys]
    obj_delta = float(vals_obj[-1]) - float(vals_obj[1])
    print(f"{'Obj Bias ' + p_name:<42} | " + " | ".join([f"{v:^12}" for v in vals_obj]) + f" | {obj_delta:+.4f}")

    reg_k = f"head.reg_preds.{p_idx}.bias"
    def get_reg_b(sd):
        t = get_tensor(sd, reg_k)
        return t.float().mean().item() if t is not None else 0.0
    vals_reg = [f"{get_reg_b(data[k].get('model', {})):.4f}" for k in keys]
    reg_delta = float(vals_reg[-1]) - float(vals_reg[1])
    print(f"{'Reg Bias ' + p_name + ' (Mean Exp)':<42} | " + " | ".join([f"{v:^12}" for v in vals_reg]) + f" | {reg_delta:+.4f}")

print("====================================================================================================================")
EOF

# Display the synthesized trajectory table in console
cat "${SYNTHESIS_REPORT}"

print_header "STEP 2 STATIC FORENSIC AUDIT COMPLETED"
echo -e "Granular logs stored in:           ${BOLD}${GREEN}${OUT_BASE}/<Milestone>/${RESET}"
echo -e "Longitudinal trajectory table in:  ${BOLD}${CYAN}${SYNTHESIS_REPORT}${RESET}\n"