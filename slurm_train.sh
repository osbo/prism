#!/bin/bash
#SBATCH --job-name=prism
#SBATCH --output=logs/prism_%j.out
#SBATCH --error=logs/prism_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --partition=gpu

set -euo pipefail
mkdir -p logs

module load cuda miniforge
source activate fluid

cd "$(dirname "$0")"

# Add ``--resume`` (or ``--resume /path/to.pt``) to continue; omit for a fresh run (deletes model.pt).
python train.py \
    --data_root  /home/osbo/orcd/pool/omniobject3d/extracted \
    --batch_size 4 \
    --n_epochs   100 \
    --image_size 256
