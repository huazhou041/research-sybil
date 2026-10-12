#!/bin/bash
#SBATCH --job-name=tangerine3d_preprocess
#SBATCH --output=logs/preprocess_%A_%a.out
#SBATCH --error=logs/preprocess_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -euo pipefail

# TANGERINE-exact preprocessing: 256^3 dynamic spacing, BSpline, raw HU saved.
# (Dataset loader applies clip [-1200,800] + fixed norm at load time.)

mkdir -p logs

DATA_ROOT="/gpfs/scratch/zhouh05/aris/sybil_saliency_202408/data"
OUTPUT_ROOT="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_tan_syb/tangerine"

JOB_ID=$SLURM_ARRAY_TASK_ID
TOTAL_JOBS=10   # MUST match --array size

module load anaconda3/gpu/new
source $(conda info --base)/etc/profile.d/conda.sh
conda activate transformer

echo "Starting job $JOB_ID of $TOTAL_JOBS"
echo "Data root:   $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"

python -u preprocess_tangerine3d.py \
    --data_root   "$DATA_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --target_size 256 256 256 \
    --intensity_min -1200 \
    --intensity_max 800 \
    --interpolator bspline \
    --format nifti \
    --job_id "$JOB_ID" \
    --total_jobs "$TOTAL_JOBS"

echo "Job $JOB_ID completed"

# After ALL array jobs finish, merge logs with:
#   python -u preprocess_tangerine3d.py --output_root /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_tan_syb/tangerine --total_jobs 10 --merge_logs
