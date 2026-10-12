#!/bin/bash
#SBATCH --job-name=roi_v15_gradcam_k50
#SBATCH --output=logs/roi_v15_gradcam_k50_%A_%a.out
#SBATCH --error=logs/roi_v15_gradcam_k50_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=03:00:00
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
export HF_HOME="$A/_hfcache/roi_occ_k50_seed${SEED}"
HFMOD="$HF_HOME/modules/transformers_modules/Pillar0-ChestCT"
mkdir -p "$HFMOD"
cp -f "$M/checkpoints/Pillar0-ChestCT"/*.py "$HFMOD"/
touch "$HFMOD/__init__.py"
cd "$M/code/pillar-finetune"
python -u "$A/code/roi_occlusion_v15.py" \
  --seed "$SEED" --top-k 50 \
  --csv "$A/_cancer_axial_cohort_v15.csv" \
  --out-root "$A/20260803/raw/roi_occlusion_v15" \
  --num-workers 4
echo "EXIT=$?"
