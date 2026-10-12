#!/bin/bash
#SBATCH --job-name=sybil_predict_ensemble
#SBATCH --output=logs/sybil_predict_ensemble_%j.out
#SBATCH --error=logs/sybil_predict_ensemble_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=a100_short,a100_long
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

set -euo pipefail

export PYTHONWARNINGS=ignore
mkdir -p logs

BASE_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct"
SCRIPT_DIR="$BASE_DIR/models/tangerine_6yrs_20260615/split2"
SYBIL_IMAGES="$BASE_DIR/data/images_3d_tan_syb/sybil"
DATASET_DIR="$SCRIPT_DIR/dataset_splits"
PRETRAINED="$SCRIPT_DIR/sybil/pretrained"
RUN_DIR="$SCRIPT_DIR/outputs/sybil_predict_only_ensemble_$(date +%Y%m%d_%H%M%S)"

CKPTS="${PRETRAINED}/28a7cd44f5bcd3e6cc760b65c7e0d54d.ckpt,${PRETRAINED}/56ce1a7d241dc342982f5466c4a9d7ef.ckpt,${PRETRAINED}/624407ef8e3a2a009f9fa51f9846fe9a.ckpt,${PRETRAINED}/64a91b25f84141d32852e75a3aec7305.ckpt,${PRETRAINED}/65fd1f04cb4c5847d86a9ed8ba31ac1a.ckpt"

mkdir -p "$SCRIPT_DIR/logs" "$RUN_DIR"

module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/new/etc/profile.d/conda.sh
conda activate transformer
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/gpfs/home/zhouh05/.conda/envs/transformer/lib:/gpfs/share/apps/anaconda3/gpu/new/envs/transformer/lib:${LD_LIBRARY_PATH:-}
unset CUDA_HOME
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] && export CUDA_VISIBLE_DEVICES=0
cd "$SCRIPT_DIR"

python -u predict_sybil_ensemble.py \
    --dataset_dir "$DATASET_DIR" \
    --images_dir "$SYBIL_IMAGES" \
    --output_dir "$RUN_DIR" \
    --sybil_checkpoints "$CKPTS" \
    --batch_size 2 --num_workers 8 --seed 20260627

echo "Sybil ensemble predict-only complete: $RUN_DIR"
