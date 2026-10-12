#!/bin/bash
#SBATCH --job-name=roi_occ_v1_full
#SBATCH --output=logs/roi_occ_v1_full_%A_%a.out
#SBATCH --error=logs/roi_occ_v1_full_%A_%a.err
#SBATCH --array=0-19
#SBATCH --time=02:30:00
#SBATCH --partition=gl40s_short,gl40s_long
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
set -uo pipefail

# Full-cohort roi_occlusion_v1 (see roi_occlusion_v1.py's module docstring). All 5
# checkpoints are loaded and run together per scan (unlike v15's one-job-per-seed
# array), so sharding here is by SCAN, 20-way -- smoke-measured 6.02s/scan implies
# ~24.3h for the full 14,549-scan cohort in one job; 20-way brings each shard to
# ~1.2h, well inside this time limit with margin. Resumable: process_scan's caller
# skips any ct_id whose ensemble/{ct_id}.npz already exists, so re-submitting the
# same --job-id/--total-jobs after a partial run only does the remaining work.
A=/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/sybil_v15_20260731/attention
JOB_ID=$SLURM_ARRAY_TASK_ID
TOTAL_JOBS=20

source activate transformer
export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if   [ -n "${SLURM_JOB_GPUS:-}" ];  then export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
  elif [ -n "${SLURM_STEP_GPUS:-}" ]; then export CUDA_VISIBLE_DEVICES="$SLURM_STEP_GPUS"
  else free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', *' '$2<500{print $1;exit}'); [ -n "$free" ] && export CUDA_VISIBLE_DEVICES="$free"; fi
fi
echo "job_id=$JOB_ID/$TOTAL_JOBS  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

cd "$A/code"
# Reuses v15's cohort_test.csv purely as a ct_id list (the same 14,549-scan
# modelling cohort, union of train+val+test) -- its image_paths column is for
# v15's RVE volumes and is ignored here; roi_occlusion_v1.py resolves each
# ct_id's own .nii.gz via NII_DIR itself. Keeps both attention pipelines
# scored over the identical cohort without a redundant CSV.
python -u roi_occlusion_v1.py \
  --csv "$A/../v15_inference/cv_eval/cohort_test.csv" \
  --out-root "$A/raw/roi_occlusion_v1" \
  --job-id "$JOB_ID" \
  --total-jobs "$TOTAL_JOBS"
echo "EXIT=$?"
