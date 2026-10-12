#!/bin/bash
#SBATCH --job-name=roi_occ_v15_full
#SBATCH --output=logs/roi_occ_v15_full_%A_%a.out
#SBATCH --error=logs/roi_occ_v15_full_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=20:00:00
#SBATCH --partition=gl40s_short,gl40s_long
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
set -uo pipefail
A=/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/sybil_v15_20260731/attention
M=/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/sybil_v15_20260731/v15_inference
SEED=$SLURM_ARRAY_TASK_ID
source activate transformer2
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if   [ -n "${SLURM_JOB_GPUS:-}" ];  then export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
  elif [ -n "${SLURM_STEP_GPUS:-}" ]; then export CUDA_VISIBLE_DEVICES="$SLURM_STEP_GPUS"
  else free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', *' '$2<500{print $1;exit}'); [ -n "$free" ] && export CUDA_VISIBLE_DEVICES="$free"; fi
fi
echo "seed=$SEED CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
export HF_HOME="$A/_hfcache/roi_occ_seed${SEED}"
HFMOD="$HF_HOME/modules/transformers_modules/Pillar0-ChestCT"
mkdir -p "$HFMOD"
cp -f "$M/checkpoints/Pillar0-ChestCT"/*.py "$HFMOD"/
[ -f "$HFMOD/__init__.py" ] || touch "$HFMOD/__init__.py"
OUTDIR="$A/raw/roi_occlusion_v15/seed${SEED}"
N_DONE=0
[ -d "$OUTDIR" ] && N_DONE=$(ls "$OUTDIR" 2>/dev/null | wc -l)
if [ "$N_DONE" -ge 14549 ]; then
  echo "SKIP (already done): $N_DONE npz in $OUTDIR"
  exit 0
fi
cd "$M/code/pillar-finetune"
python -u "$A/code/roi_occlusion_v15.py" \
  --seed "$SEED" \
  --csv "$M/cv_eval/cohort_test.csv" \
  --out-root "$A/raw/roi_occlusion_v15" \
  --num-workers 8
echo "EXIT=$?"
