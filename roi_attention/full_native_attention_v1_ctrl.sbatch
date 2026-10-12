#!/bin/bash
#SBATCH --job-name=native_attn_v1_ctrl
#SBATCH --output=logs/native_attn_v1_ctrl_%j.out
#SBATCH --error=logs/native_attn_v1_ctrl_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gl40s_short,gl40s_long
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
set -uo pipefail
A=/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/sybil_v15_20260731/attention
source activate transformer
export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if   [ -n "${SLURM_JOB_GPUS:-}" ];  then export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
  elif [ -n "${SLURM_STEP_GPUS:-}" ]; then export CUDA_VISIBLE_DEVICES="$SLURM_STEP_GPUS"
  else free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', *' '$2<500{print $1;exit}'); [ -n "$free" ] && export CUDA_VISIBLE_DEVICES="$free"; fi
fi
cd "$A/code"
python -u roi_native_attention_v1.py --top-k 50 \
  --csv "$A/_control_axial_ct_ids.csv" \
  --out-root "$A/20260803/raw/native_attn_v1"
echo "EXIT=$?"
