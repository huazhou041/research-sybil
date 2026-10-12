#!/bin/bash
#SBATCH --job-name=sybil_predict_only
#SBATCH --output=logs/sybil_predict_%j.out
#SBATCH --error=logs/sybil_predict_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=a100_short,a100_long
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail

export PYTHONWARNINGS=ignore
mkdir -p logs

BASE_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct"
SCRIPT_DIR="$BASE_DIR/models/tangerine_6yrs_20260615/split2"
SYBIL_IMAGES="$BASE_DIR/data/images_3d_tan_syb/sybil"
DATASET_DIR="$SCRIPT_DIR/dataset_splits"
SYBIL_CKPT="$SCRIPT_DIR/sybil/pretrained/28a7cd44f5bcd3e6cc760b65c7e0d54d.ckpt"
RUN_DIR="$SCRIPT_DIR/outputs/sybil_predict_only_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$SCRIPT_DIR/logs" "$RUN_DIR"

module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/new/etc/profile.d/conda.sh
conda activate transformer
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/gpfs/home/zhouh05/.conda/envs/transformer/lib:/gpfs/share/apps/anaconda3/gpu/new/envs/transformer/lib:${LD_LIBRARY_PATH:-}
unset CUDA_HOME
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] && export CUDA_VISIBLE_DEVICES=0
cd "$SCRIPT_DIR"

python predict_sybil.py \
    --dataset_dir      "$DATASET_DIR" \
    --images_dir       "$SYBIL_IMAGES" \
    --output_dir       "$RUN_DIR" \
    --sybil_checkpoint "$SYBIL_CKPT" \
    --batch_size 2 --num_workers 8 --seed 20260627

echo "Sybil predict-only complete: $RUN_DIR"
