#!/bin/bash
# Drive baseline_sweep.py one cell per python invocation. Each cell writes to
# the same CSV; the per-cell mode auto-skips already-done cells via
# _resume_state(). Intentionally no `set -e` so a SIGKILL on one cell doesn't
# stop the loop.
#
# Why one-process-per-cell: heavy 5-fold cluster_k5 cells on bigger models
# (resnext101_32x8d_wsl etc.) can hit ~30 GB RAM during ridge fitting and
# macOS jetsam will SIGKILL the process. Spawning a fresh python per cell
# resets memory pressure between cells so one crash doesn't terminate the sweep.
#
# Usage:  bash brainscore_vision/benchmarks/laion_fmri/baselines/sweep_per_cell.sh
# Resume: simply re-run; cells already in the CSV are skipped within ~3-5s of
#         python startup.

source ~/.zshrc 2>/dev/null || conda activate 2>/dev/null || true
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate vision-2026 2>/dev/null || true

# Resolve this script's directory so we can invoke baseline_sweep.py via its
# absolute path regardless of where the user runs us from.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SWEEP_PY="$SCRIPT_DIR/baseline_sweep.py"
if [ ! -f "$SWEEP_PY" ]; then
    echo "ERROR: baseline_sweep.py not found next to this script ($SCRIPT_DIR)" >&2
    exit 1
fi

MODELS=(
    "alexnet"
    "alexnet_random"
    "resnet50_tutorial"
    "convnext_tiny_imagenet1K_torchvisionV1"
    "resnext101_32x8d_wsl"
    "convnext_base_fb_in22k_ft_in1k"
)
REGIONS=("V1" "V2" "V4" "IT")
SPLITS=("tau" "ood")
FAMILIES=("LAION_fMRI" "LAION_fMRI_persubject")

# 20 headline benchmarks per model: 16 ridge + 4 shared RSA.
CELLS=()
for M in "${MODELS[@]}"; do
    for FAM in "${FAMILIES[@]}"; do
        for R in "${REGIONS[@]}"; do
            for S in "${SPLITS[@]}"; do
                CELLS+=("$M ${FAM}.${R}-${S}-ridge")
            done
        done
    done
    for R in "${REGIONS[@]}"; do
        CELLS+=("$M LAION_fMRI.${R}-rdm-pearson")
    done
done

TOTAL=${#CELLS[@]}
echo "Per-cell sweep: ${TOTAL} cells (${#MODELS[@]} models x 20 benchmarks)"
echo "Started: $(date)"

I=0
for CELL in "${CELLS[@]}"; do
    I=$((I+1))
    read -r M B <<<"${CELL}"
    python -u "$SWEEP_PY" "$M" "$B" "$I" "$TOTAL"
done

echo "Done: $(date)"
