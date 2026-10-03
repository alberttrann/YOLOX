#!/usr/bin/env bash
# ==============================================================================
# TDE-YOLOX v3.2 Master Forensic Probing Suite
# Cross-Domain ACDC Adverse Scanner & Dynamic Image Auto-Resolver
# ==============================================================================

set -eo pipefail

export PYTHONIOENCODING="utf-8"
export PYTHONUTF8=1
export LANG="en_US.UTF-8"

PYTHON_BIN="python"
EXP_FILE="exps/default/tde_yolox_s.py"

CKPT_DIR="YOLOX_outputs/tde_yolox_s"
PROBE_OUTPUT_DIR="./probe_results"
LOG_DIR="${PROBE_OUTPUT_DIR}/logs"

mkdir -p "${PROBE_OUTPUT_DIR}" "${LOG_DIR}"

MASTER_LOG="${LOG_DIR}/master_probe_session.log"
echo "=== TDE-YOLOX FORENSIC PROBE SESSION START: $(date) ===" > "${MASTER_LOG}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

# Resolve active checkpoint
RESOLVED_CKPT=""
if [[ -n "${2:-}" && "$2" == *.pth ]]; then
    RESOLVED_CKPT="$2"
elif [[ -n "${1:-}" && "$1" == *.pth ]]; then
    RESOLVED_CKPT="$1"
fi

if [ -z "$RESOLVED_CKPT" ]; then
    if [ -d "$CKPT_DIR" ]; then
        HIGHEST_EPOCH=$(ls -1 "$CKPT_DIR"/epoch_*_ckpt.pth 2>/dev/null | grep -E 'epoch_[0-9]+_ckpt\.pth' | sort -V | tail -n 1 || true)
        if [ -n "$HIGHEST_EPOCH" ] && [ -f "$HIGHEST_EPOCH" ]; then
            RESOLVED_CKPT="$HIGHEST_EPOCH"
        elif [ -f "$CKPT_DIR/best_ckpt.pth" ]; then
            RESOLVED_CKPT="$CKPT_DIR/best_ckpt.pth"
        elif [ -f "$CKPT_DIR/latest_ckpt.pth" ]; then
            RESOLVED_CKPT="$CKPT_DIR/latest_ckpt.pth"
        fi
    fi
fi

CKPT_PATH="${RESOLVED_CKPT:-YOLOX_outputs/tde_yolox_s/best_ckpt.pth}"

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
    echo -e "${BOLD}${CYAN}==> [COMMAND]:${RESET} ${YELLOW}$1${RESET}"
}

if [ ! -f "${CKPT_PATH}" ]; then
    echo -e "${RED}Error: Checkpoint file not found at: ${CKPT_PATH}${RESET}"
    exit 1
fi

echo -e "${BOLD}${GREEN}✔ Active Checkpoint Selected:${RESET} ${BOLD}${YELLOW}${CKPT_PATH}${RESET}"
echo -e "${BOLD}${CYAN}✔ Master Log Stream Active:${RESET}   ${BOLD}${MASTER_LOG}${RESET}"

# ------------------------------------------------------------------------------
# DYNAMIC IMAGE AUTO-RESOLVER (ACDC & BDD100K ADVERSE SPLITS)
# ------------------------------------------------------------------------------
resolve_image() {
    local domain="$1"
    ${PYTHON_BIN} -c "
import os, glob, json

domain = '${domain}'.lower()
roots = ['D:/YOLOX-3RD', 'C:/TDE-YOLOX', './']

# 1. Search ACDC dataset structure
for root in roots:
    # Pattern 1: acdc/rgb_anon/<domain>/val/*.png
    pats = [
        os.path.join(root, 'acdc', 'rgb_anon', domain, 'val', '*.png'),
        os.path.join(root, 'acdc', domain, 'val', '*.png'),
        os.path.join(root, 'acdc', '**', domain, '**', '*.png'),
        os.path.join(root, 'acdc', '**', domain, '**', '*.jpg'),
    ]
    for p in pats:
        matches = glob.glob(p, recursive=True)
        if matches:
            print(os.path.abspath(matches[0]).replace('\\\\', '/'))
            exit(0)

# 2. Fallback: Search BDD100K adverse annotations
for root in roots:
    ann_path = os.path.join(root, 'bdd100k/bdd100k/bdd100k/images/annotations/tde_test_adverse_coco.json')
    img_dir = os.path.join(root, 'bdd100k/bdd100k/bdd100k/images')
    if os.path.exists(ann_path) and os.path.exists(img_dir):
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)
            for img in data.get('images', []):
                w = img.get('weather', '').lower()
                if (domain == 'fog' and w == 'foggy') or (domain == 'snow' and w == 'snowy') or (domain == 'rain' and w == 'rainy'):
                    full_p = os.path.join(img_dir, img['file_name'])
                    if os.path.exists(full_p):
                        print(os.path.abspath(full_p).replace('\\\\', '/'))
                        exit(0)
        except Exception:
            pass

# 3. Clean daylight fallback
clean_fallback = 'D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/100k/val/b1c66a42-6f7d68ca.jpg'
if os.path.exists(clean_fallback):
    print(clean_fallback)
    exit(0)

print('')
"
}

# ------------------------------------------------------------------------------
# SUITE A: STATIC CHECKPOINT FORENSICS (probe_pth.py)
# ------------------------------------------------------------------------------
run_static_probes() {
    print_header "STAGE A: STATIC CHECKPOINT FORENSICS (probe_pth.py)"

    print_step "${PYTHON_BIN} probe_pth.py -c ${CKPT_PATH}"
    ${PYTHON_BIN} probe_pth.py -c "${CKPT_PATH}"

    print_step "${PYTHON_BIN} probe_pth.py -c ${CKPT_PATH} -v --filter memory_banks"
    ${PYTHON_BIN} probe_pth.py -c "${CKPT_PATH}" -v --filter "memory_banks"

    print_step "${PYTHON_BIN} probe_pth.py -c ${CKPT_PATH} -v --filter dark2"
    ${PYTHON_BIN} probe_pth.py -c "${CKPT_PATH}" -v --filter "dark2"

    print_step "${PYTHON_BIN} probe_pth.py -c ${CKPT_PATH} -v --filter bqsa"
    ${PYTHON_BIN} probe_pth.py -c "${CKPT_PATH}" -v --filter "bqsa"

    print_step "${PYTHON_BIN} probe_pth.py -c ${CKPT_PATH} -v --filter attnres"
    ${PYTHON_BIN} probe_pth.py -c "${CKPT_PATH}" -v --filter "attnres"

    print_step "${PYTHON_BIN} probe_pth.py -c ${CKPT_PATH} -v --filter preds"
    ${PYTHON_BIN} probe_pth.py -c "${CKPT_PATH}" -v --filter "preds"
}

# ------------------------------------------------------------------------------
# SUITE B: CLEAN DAYLIGHT BASELINE PROBE
# ------------------------------------------------------------------------------
run_live_probes() {
    print_header "STAGE B: CLEAN DAYLIGHT BASELINE PROBE (probe_inference.py)"

    local clean_img
    clean_img=$(resolve_image "clean")
    if [ -z "${clean_img}" ] || [ ! -f "${clean_img}" ]; then
        echo -e "${YELLOW}Warning: Clean sample image not found. Skipping Stage B.${RESET}"
        return
    fi

    # Standard Detection Probe (conf=0.25, FP16)
    print_step "${PYTHON_BIN} probe_inference.py -f ${EXP_FILE} -c ${CKPT_PATH} -i ${clean_img} --domain clean --conf 0.25 --fp16"
    ${PYTHON_BIN} probe_inference.py \
        -f "${EXP_FILE}" \
        -c "${CKPT_PATH}" \
        -i "${clean_img}" \
        --domain "clean" \
        --conf 0.25 \
        --nms 0.65 \
        --fp16 \
        --save_dir "${PROBE_OUTPUT_DIR}/standard" \
        --device "cuda"

    # Low-Confidence Benchmark Audit (conf=0.001, FP16)
    print_step "${PYTHON_BIN} probe_inference.py -f ${EXP_FILE} -c ${CKPT_PATH} -i ${clean_img} --domain clean --conf 0.001 --fp16"
    ${PYTHON_BIN} probe_inference.py \
        -f "${EXP_FILE}" \
        -c "${CKPT_PATH}" \
        -i "${clean_img}" \
        --domain "clean" \
        --conf 0.001 \
        --nms 0.65 \
        --fp16 \
        --save_dir "${PROBE_OUTPUT_DIR}/low_conf_audit" \
        --device "cuda"
}

# ------------------------------------------------------------------------------
# SUITE C: 4-DOMAIN ADVERSE WEATHER SCAN (FOG, SNOW, RAIN, NIGHT)
# ------------------------------------------------------------------------------
run_multi_weather_probe() {
    print_header "STAGE C: 4-DOMAIN ADVERSE WEATHER SCAN (ACDC / BDD ADVERSE)"

    DOMAINS=("fog" "snow" "rain" "night")
    PROBED_COUNT=0

    for dom in "${DOMAINS[@]}"; do
        echo -e "\n${BOLD}${MAGENTA}==> Resolving image for adverse condition: [${dom^^}]...${RESET}"
        img_resolved=$(resolve_image "$dom")

        if [ -n "$img_resolved" ] && [ -f "$img_resolved" ]; then
            echo -e "    ${GREEN}Found:${RESET} ${img_resolved}"
            print_step "${PYTHON_BIN} probe_inference.py -f ${EXP_FILE} -c ${CKPT_PATH} -i ${img_resolved} --domain ${dom} --conf 0.20 --fp16"
            
            ${PYTHON_BIN} probe_inference.py \
                -f "${EXP_FILE}" \
                -c "${CKPT_PATH}" \
                -i "$img_resolved" \
                --domain "$dom" \
                --conf 0.20 \
                --nms 0.65 \
                --fp16 \
                --save_dir "${PROBE_OUTPUT_DIR}/adverse/${dom}" \
                --device "cuda"
            
            PROBED_COUNT=$((PROBED_COUNT + 1))
        else
            echo -e "    ${RED}Could not locate test image for [${dom}]. Check dataset folder paths.${RESET}"
        fi
    done

    # Generate Cross-Domain Summary Table from Telemetry Tokens
    if [ "$PROBED_COUNT" -gt 0 ]; then
        echo -e "\n${BOLD}${WHITE}================================================================================${RESET}"
        echo -e "${BOLD}${WHITE}   CROSS-DOMAIN INFERENCE FORENSIC AUDIT SUMMARY                                ${RESET}"
        echo -e "${BOLD}${WHITE}================================================================================${RESET}"
        ${PYTHON_BIN} -c "
import glob, sys

log_file = '${MASTER_LOG}'
records = []

if os.path.exists(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('__TELEMETRY__|'):
                parts = line.strip().split('|')[1:]
                records.append(parts)

if records:
    header = ['Domain', 'Haze Idx', 'H(s) nats', 'TTT Mask%', 'TTT Loss', 'Engram Gate', 'Full Dets', 'No-TTT', 'No-Engram']
    rows = []
    for r in records:
        rows.append(r)
    
    col_w = [10, 10, 12, 12, 12, 12, 10, 10, 10]
    sep = '+-' + '-+-'.join(['-' * w for w in col_w]) + '-+'
    hdr_line = '| ' + ' | '.join([f'{header[i]:<{col_w[i]}}' for i in range(len(header))]) + ' |'
    
    print(sep)
    print(hdr_line)
    print(sep)
    for r in rows:
        line_str = '| ' + ' | '.join([f'{r[i]:<{col_w[i]}}' for i in range(min(len(r), len(col_w)))]) + ' |'
        print(line_str)
    print(sep)
"
    fi
}

MODE="${1:-all}"
if [[ "$MODE" == *.pth ]]; then
    MODE="all"
fi

case "$MODE" in
    static) run_static_probes ;;
    live)   run_live_probes ;;
    multi)  run_multi_weather_probe ;;
    all)
        run_static_probes
        run_live_probes
        run_multi_weather_probe
        ;;
    *)
        echo "Usage: $0 {all|static|live|multi} [path/to/checkpoint.pth]"
        exit 1
        ;;
esac

print_header "ALL FORENSIC PROBING COMMANDS COMPLETED SUCCESSFULLY"
echo -e "Master session log saved to:        ${BOLD}${GREEN}${MASTER_LOG}${RESET}"
echo -e "Visual diagnostic dashboards in:    ${BOLD}${PROBE_OUTPUT_DIR}${RESET}\n"