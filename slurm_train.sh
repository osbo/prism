#!/bin/bash
#SBATCH --job-name=prism
#SBATCH --output=logs/prism_%j.out
#SBATCH --error=logs/prism_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # one task per GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu              # adjust to the ORCD partition name

mkdir -p logs

# Load modules (adjust to ORCD Engaging module names)
module purge
module load cuda/12.1
module load python/3.11

# Activate your virtualenv / conda env
source ~/.venvs/prism/bin/activate
# or: conda activate prism

DATA_ROOT=/orcd/pool/007/osbo/omniobject3d
OUTPUT_DIR=/orcd/pool/007/osbo/prism_runs

# torchrun handles MASTER_ADDR / MASTER_PORT for single-node DDP
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    /orcd/pool/007/osbo/PRISM/train.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --n_epochs 100 \
    --n_rays_train 512
