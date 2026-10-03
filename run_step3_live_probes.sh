#!/usr/bin/env bash
# ==============================================================================
# TDE-YOLOX v3.2 Step 3: Longitudinal Live Inference X-Ray Probes
# 5 Milestones x 5 Weather Scenes = 25 Tapped Forward Passes
# Path-Aligned for D:/ACDC/rgb_anon/
# ==============================================================================

set -eo pipefail

export PYTHONIOENCODING="utf-8"
export PYTHONUTF8=1
export LANG="en_US.UTF-8"

# 1. PYTHON AUTO-DETECTION
if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/Scripts/python.exe" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/Scripts/python.exe"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
elif [ -f ".venv/Scripts/python" ]; then
    PYTHON_BIN=".venv/Scripts/python"
else
    PYTHON_BIN="python"
fi

EXP_FILE="exps/default/tde_yolox_s.py"
CKPT_DIR="YOLOX_outputs/tde_yolox_s"
OUT_BASE="./probe_results/step3_live_probes"

mkdir -p "${OUT_BASE}"
LOG_FILE="${OUT_BASE}/step3_master.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
RED="\033[31m"
RESET="\033[0m"

echo -e "\n${BOLD}${CYAN}================================================================================${RESET}"
echo -e "${BOLD}${CYAN}   TDE-YOLOX STEP 3: LONGITUDINAL LIVE X-RAY PROBE (25 PASSES)                  ${RESET}"
echo -e "${BOLD}${CYAN}================================================================================${RESET}\n"

# 2. RESOLVE CHECKPOINTS
resolve_milestone() {
    local primary="$1"
    local fallback="$2"
    if [ -f "${CKPT_DIR}/${primary}" ]; then
        echo "${CKPT_DIR}/${primary}"
    elif [ -f "${CKPT_DIR}/${fallback}" ]; then
        echo "${CKPT_DIR}/${fallback}"
    fi
}

M1=$(resolve_milestone "epoch_10_ckpt.pth" "epoch_10_ckpt.pth")
M2=$(resolve_milestone "epoch_24_ckpt.pth" "best_ckpt.pth")
M3=$(resolve_milestone "epoch_40_ckpt.pth" "epoch_38_ckpt.pth")
M4=$(resolve_milestone "epoch_64_ckpt.pth" "epoch_65_ckpt.pth")
M5=$(resolve_milestone "epoch_78_ckpt.pth" "latest_ckpt.pth")

TAGS=("M1_Warmup_E10" "M2_Peak_E24" "M3_Inflection_E40" "M4_SagFloor_E64" "M5_Terminal_E78")
PATHS=("$M1" "$M2" "$M3" "$M4" "$M5")

# 3. ROBUST IMAGE RESOLVER (Explicitly covers D:/ACDC/rgb_anon/)
resolve_image() {
    local domain="$1"
    ${PYTHON_BIN} -c "
import os, glob

domain = '${domain}'.lower()

if domain == 'clean':
    clean_candidates = [
        'D:/YOLOX-3RD/bdd100k/bdd100k/bdd100k/images/100k/val/b1c66a42-6f7d68ca.jpg',
        'D:/YOLOX-3RD/bdd100k/bdd100k/images/100k/val/b1c66a42-6f7d68ca.jpg'
    ]
    for c in clean_candidates:
        if os.path.exists(c):
            print(c)
            exit(0)

# Check D:/ACDC/rgb_anon/ directly as shown in File Explorer
search_roots = [
    f'D:/ACDC/rgb_anon/{domain}',
    f'D:/ACDC/{domain}',
    f'D:/YOLOX-3RD/acdc/rgb_anon/{domain}',
    f'D:/YOLOX-3RD/acdc/{domain}'
]

for sdir in search_roots:
    if os.path.exists(sdir):
        patterns = [
            os.path.join(sdir, 'val', '**', '*.png'),
            os.path.join(sdir, '**', '*.png'),
            os.path.join(sdir, '**', '*.jpg')
        ]
        for pat in patterns:
            matches = glob.glob(pat, recursive=True)
            if matches:
                # Prefer val images if present
                val_m = [m for m in matches if 'val' in m.replace('\\\\', '/').split('/')]
                target = val_m[0] if val_m else matches[0]
                print(os.path.abspath(target).replace('\\\\', '/'))
                exit(0)

print('')
"
}

DOMAINS=("clean" "fog" "snow" "rain" "night")
declare -A DOMAIN_IMGS

echo -e "${BOLD}Resolving Test Domain Images:${RESET}"
for dom in "${DOMAINS[@]}"; do
    img=$(resolve_image "$dom")
    if [ -n "$img" ] && [ -f "$img" ]; then
        DOMAIN_IMGS["$dom"]="$img"
        printf "  • %-10s -> ${GREEN}%s${RESET}\n" "[${dom^^}]" "$img"
    else
        echo -e "  • ${RED}[${dom^^}] Image NOT found on disk!${RESET}"
    fi
done

# 4. EXECUTE 25-PASS AUDIT MATRIX
for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"
    ckpt="${PATHS[$i]}"
    echo -e "\n${BOLD}${YELLOW}--------------------------------------------------------------------------------${RESET}"
    echo -e "${BOLD}${YELLOW}  Probing Milestone: [${tag}] (${ckpt})${RESET}"
    echo -e "${BOLD}${YELLOW}--------------------------------------------------------------------------------${RESET}"

    for dom in "${DOMAINS[@]}"; do
        img="${DOMAIN_IMGS[$dom]}"
        if [ -z "$img" ] || [ ! -f "$img" ]; then
            continue
        fi

        save_target="${OUT_BASE}/${tag}/${dom}"
        mkdir -p "${save_target}"

        echo -e "  ${CYAN}==> Forward X-Ray:${RESET} ${tag} | Domain: [${dom^^}]"
        ${PYTHON_BIN} probe_inference.py \
            -f "${EXP_FILE}" \
            -c "${ckpt}" \
            -i "${img}" \
            --domain "${dom}" \
            --conf 0.20 \
            --nms 0.65 \
            --fp16 \
            --save_dir "${save_target}" \
            --device "cuda"
    done
done

# 5. SYNTHESIZE CROSS-MILESTONE TELEMETRY TABLE
echo -e "\n${BOLD}${WHITE}================================================================================${RESET}"
echo -e "${BOLD}${WHITE}   STEP 3 SYNTHESIS: LONGITUDINAL CROSS-DOMAIN TELEMETRY MATRIX                 ${RESET}"
echo -e "${BOLD}${WHITE}================================================================================${RESET}"

${PYTHON_BIN} - << 'EOF'
import os

log_file = "./probe_results/step3_live_probes/step3_master.log"
records = []

if os.path.exists(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        curr_tag = ""
        for line in f:
            if "Probing Milestone: [" in line:
                curr_tag = line.split("[")[1].split("]")[0]
            if line.startswith("__TELEMETRY__|"):
                parts = line.strip().split('|')[1:]
                records.append([curr_tag] + parts)

if records:
    header = ['Milestone', 'Domain', 'HazeIdx', 'H(s) nats', 'Mask%', 'TTTLoss', 'Gate(P3)', 'Full', 'NoTTT', 'NoEng']
    col_w = [18, 8, 8, 10, 8, 10, 10, 6, 6, 6]
    sep = '+-' + '-+-'.join(['-' * w for w in col_w]) + '-+'
    hdr_line = '| ' + ' | '.join([f'{header[i]:<{col_w[i]}}' for i in range(len(header))]) + ' |'
    
    print(sep)
    print(hdr_line)
    print(sep)
    for r in records:
        print('| ' + ' | '.join([f'{r[i]:<{col_w[i]}}' for i in range(min(len(r), len(col_w)))]) + ' |')
    print(sep)
else:
    print("No telemetry records gathered.")
EOF

echo -e "\n${BOLD}${GREEN}✔ Step 3 Live Probing Completed! Dashboards in: ${OUT_BASE}/${RESET}\n"