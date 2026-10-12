#!/bin/bash
#SBATCH --job-name=controls_render_panels
#SBATCH --output=logs/controls_render_panels_%A_%a.out
#SBATCH --error=logs/controls_render_panels_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
set -uo pipefail
A=/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/sybil_v15_20260731/attention
source activate transformer2
export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
cd "$A/code"

declare -a FAMS=(v1        v1          v15)
declare -a VARS=(single    ensemble    x)
declare -a OUTS=(v1_single v1_ensemble v15_combined)

i=$SLURM_ARRAY_TASK_ID
if [ "${FAMS[$i]}" = "v1" ]; then
  python3 -u render_roi_panel.py --acc-list "$A/_control_axial_ct_ids.txt" --out-dir "$A/20260803/roi_images_controls/${OUTS[$i]}" --family v1 --variant "${VARS[$i]}"
else
  python3 -u render_roi_panel.py --acc-list "$A/_control_axial_ct_ids.txt" --out-dir "$A/20260803/roi_images_controls/${OUTS[$i]}" --family v15
fi
echo "EXIT=$?"
