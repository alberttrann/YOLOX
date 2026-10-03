#!/usr/bin/env bash
# ==============================================================================
# TDE-YOLOX v3.2 Step 4: 25-Run Full Benchmark Grid Matrix
# 5 Milestones x 5 Evaluation Datasets = Full Longitudinal Curve
# ==============================================================================

set -eo pipefail

export PYTHONIOENCODING="utf-8"
export PYTHONUTF8=1
export LANG="en_US.UTF-8"

if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/Scripts/python.exe" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/Scripts/python.exe"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
elif [ -f ".venv/Scripts/python" ]; then
    PYTHON_BIN=".venv/Scripts/python"
else
    PYTHON_BIN="python"
fi

CKPT_DIR="YOLOX_outputs/tde_yolox_s"
OUT_BASE="./probe_results/step4_benchmark_grid"
mkdir -p "${OUT_BASE}"

LOG_FILE="${OUT_BASE}/step4_master.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
RESET="\033[0m"

echo -e "\n${BOLD}${CYAN}================================================================================${RESET}"
echo -e "${BOLD}${CYAN}   TDE-YOLOX STEP 4: 25-RUN FULL BENCHMARK MATRIX                               ${RESET}"
echo -e "${BOLD}${CYAN}================================================================================${RESET}\n"

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

# 5 Evaluation Experiments
EXPS=(
    "exps/default/tde_yolox_s.py"      # BDD100K Clean Val
    "exps/default/test_acdc_fog.py"    # ACDC Fog
    "exps/default/test_acdc_snow.py"   # ACDC Snow
    "exps/default/test_acdc_rain.py"   # ACDC Rain
    "exps/default/test_acdc_night.py"  # ACDC Night
)
DOMAIN_NAMES=("Clean_Val" "ACDC_Fog" "ACDC_Snow" "ACDC_Rain" "ACDC_Night")

for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"
    ckpt="${PATHS[$i]}"

    for j in "${!EXPS[@]}"; do
        exp_file="${EXPS[$j]}"
        dname="${DOMAIN_NAMES[$j]}"

        if [ ! -f "$exp_file" ]; then
            echo -e "${YELLOW}Warning: Experiment file not found: ${exp_file}. Skipping.${RESET}"
            continue
        fi

        eval_out="${OUT_BASE}/${tag}/${dname}"
        mkdir -p "${eval_out}"

        echo -e "\n${BOLD}${CYAN}==> [BENCHMARK RUN]:${RESET} ${tag} on ${BOLD}${YELLOW}[${dname}]${RESET}"
        ${PYTHON_BIN} tools/eval.py \
            -f "${exp_file}" \
            -c "${ckpt}" \
            -b 16 \
            --conf 0.001 \
            --fp16 \
            --fuse \
            --experiment-name "${eval_out}"
    done
done

echo -e "\n${BOLD}${GREEN}✔ All 25 Benchmark Evaluations Completed! Logged in: ${OUT_BASE}/${RESET}\n"