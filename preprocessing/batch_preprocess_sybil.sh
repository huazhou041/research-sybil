#!/bin/bash
#SBATCH --job-name=sybil3d_preprocess
#SBATCH --output=logs/preprocess_sybil_%A_%a.out
#SBATCH --error=logs/preprocess_sybil_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -euo pipefail

# Sybil-exact preprocessing: per-slice <=2.5mm filter, -600/1500 window,
# resample 0.703x0.703x2.5, CropOrPad 256x256x200, 1-channel saved.
# (Dataset applies z-score norm + 3-channel expansion at load time.)

mkdir -p logs

DATA_ROOT="/gpfs/scratch/zhouh05/aris/sybil_saliency_202408/data"
OUTPUT_ROOT="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_tan_syb/sybil"

JOB_ID=$SLURM_ARRAY_TASK_ID
TOTAL_JOBS=10   # MUST match --array size

module load anaconda3/gpu/new
source $(conda info --base)/etc/profile.d/conda.sh
conda activate transformer

echo "Starting Sybil preprocess job $JOB_ID of $TOTAL_JOBS"
echo "Data root:   $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"

python -u preprocess_sybil3d.py \
    --data_root   "$DATA_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --max_slice_thickness 2.5 \
    --format nifti \
    --job_id "$JOB_ID" \
    --total_jobs "$TOTAL_JOBS"

echo "Job $JOB_ID completed"

# After ALL array jobs finish, merge logs:
#   python -u preprocess_sybil3d.py --output_root /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_tan_syb/sybil --total_jobs 10 --merge_logs
#
# NOTE: torchio, opencv (cv2), pydicom required in the 'transformer' env.
#       If missing:  pip install torchio opencv-python-headless pydicom
