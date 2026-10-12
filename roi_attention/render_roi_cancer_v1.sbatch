#!/bin/bash
#SBATCH --job-name=roi_cancer_render_v1
#SBATCH --output=logs/roi_cancer_render_v1_%j.out
#SBATCH --error=logs/roi_cancer_render_v1_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
set -uo pipefail
A=/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/sybil_v15_20260731/attention
source activate transformer
export PYTHONNOUSERSITE=1
cd "$A/code"
python3 -u render_roi_cancer_batch_v1.py \
  --acc-list "$A/_cancer_axial_ct_ids.txt" \
  --out-dir "$A/roi_images_v1_cancer"
echo "EXIT=$?"
