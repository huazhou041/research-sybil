#!/bin/bash
#SBATCH --job-name=prepare_split_tan_syb
#SBATCH --output=logs/prepare_split_%j.out
#SBATCH --error=logs/prepare_split_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_short,cpu_medium
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -euo pipefail

# Prepare the patient-level survival split on the COMMON cohort
# (intersection of the TANGERINE- and Sybil-preprocessed sets).
# Output splits go into THIS modelling folder's dataset_splits/.

export PYTHONWARNINGS=ignore
mkdir -p logs

BASE_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct"
SCRIPT_DIR="$BASE_DIR/models/tangerine_6yrs_20260615/split2"
DATA_DIR="$BASE_DIR/data/images_3d_tan_syb"
TAN_IMAGES="$DATA_DIR/tangerine"     # TANGERINE-preprocessed (256^3 raw HU)
SYB_IMAGES="$DATA_DIR/sybil"         # Sybil-preprocessed (256x256x200 windowed)
METADATA_CSV="$BASE_DIR/lungct_with_mrn_anonacc.csv"
OUTPUT_DIR="$SCRIPT_DIR/dataset_splits"

echo "Preparing split"
echo "  tangerine set: $TAN_IMAGES"
echo "  sybil set:     $SYB_IMAGES"
echo "  metadata:      $METADATA_CSV"
echo "  output:        $OUTPUT_DIR"

module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/new/etc/profile.d/conda.sh
conda activate transformer
export PYTHONNOUSERSITE=1
cd "$SCRIPT_DIR"

python prepare_survival_dataset.py \
    --metadata_csv   "$METADATA_CSV" \
    --images_dir     "$TAN_IMAGES" \
    --intersect_with "$SYB_IMAGES" \
    --output_dir     "$OUTPUT_DIR" \
    --max_followup 6 \
    --seed 20260627

echo ""
echo "Split written to: $OUTPUT_DIR"
echo "  train.csv / val.csv / test.csv / common_ct_ids.txt / config.json"
echo "Both TANGERINE and Sybil training read these SAME split CSVs."
