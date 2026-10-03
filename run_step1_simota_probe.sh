#!/usr/bin/env bash
# ==============================================================================
# TDE-YOLOX v3.2 Step 1: SimOTA Dynamic Scale-Allocation Forensic Probe
# Quantifies Foreground Assignment Counts across P3, P4, and P5
# ==============================================================================

set -eo pipefail

export PYTHONIOENCODING="utf-8"
export PYTHONUTF8=1
export LANG="en_US.UTF-8"

# 1. PYTHON AUTO-DETECTION
if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/Scripts/python.exe" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/Scripts/python.exe"
elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
elif [ -f ".venv/Scripts/python" ]; then
    PYTHON_BIN=".venv/Scripts/python"
else
    PYTHON_BIN="python"
fi

EXP_FILE="exps/default/tde_yolox_s.py"
CKPT_DIR="YOLOX_outputs/tde_yolox_s"
OUT_DIR="./probe_results/simota_audit"
mkdir -p "${OUT_DIR}"

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
RED="\033[31m"
RESET="\033[0m"

echo -e "\n${BOLD}${CYAN}================================================================================${RESET}"
echo -e "${BOLD}${CYAN}   TDE-YOLOX STEP 1: SimOTA ANCHOR ALLOCATION PROBE (P3 vs. P4 vs. P5)           ${RESET}"
echo -e "${BOLD}${CYAN}================================================================================${RESET}\n"

# Resolve checkpoint
RESOLVED_CKPT=""
if [[ -n "${1:-}" && "$1" == *.pth ]]; then
    RESOLVED_CKPT="$1"
fi

if [ -z "$RESOLVED_CKPT" ]; then
    if [ -f "${CKPT_DIR}/epoch_24_ckpt.pth" ]; then
        RESOLVED_CKPT="${CKPT_DIR}/epoch_24_ckpt.pth"
    elif [ -f "${CKPT_DIR}/best_ckpt.pth" ]; then
        RESOLVED_CKPT="${CKPT_DIR}/best_ckpt.pth"
    elif [ -f "${CKPT_DIR}/latest_ckpt.pth" ]; then
        RESOLVED_CKPT="${CKPT_DIR}/latest_ckpt.pth"
    fi
fi

if [ ! -f "${RESOLVED_CKPT}" ]; then
    echo -e "${RED}Error: Checkpoint file not found: ${RESOLVED_CKPT}${RESET}"
    exit 1
fi

echo -e "Using Python Binary:   ${BOLD}${GREEN}${PYTHON_BIN}${RESET}"
echo -e "Auditing Checkpoint:   ${BOLD}${YELLOW}${RESOLVED_CKPT}${RESET}"
echo -e "Output Directory:      ${BOLD}${OUT_DIR}${RESET}\n"

# Run 100-batch validation probe (evaluates 1,600 real images)
${PYTHON_BIN} probe_simota_allocation.py \
    -f "${EXP_FILE}" \
    -c "${RESOLVED_CKPT}" \
    --split "val" \
    --num_batches 100 \
    -b 16 \
    --save_dir "${OUT_DIR}" \
    --device "cuda"

echo -e "\n${BOLD}${GREEN}✔ SimOTA Allocation Audit Complete! Reports saved in: ${OUT_DIR}/${RESET}\n"