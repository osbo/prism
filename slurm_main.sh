#!/bin/bash
#SBATCH --job-name=prism_main
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --partition=mit_normal_gpu

set -euo pipefail
mkdir -p logs

module load cuda miniforge
source activate prism

cd "$SLURM_SUBMIT_DIR"

# Full model, 30 epochs.  Evaluate on 100 test objects to keep eval time reasonable.
python run_experiment.py \
    --exp_name full_model \
    --n_epochs 30 \
    --n_eval_objects 100

echo "Main experiment done.  Results in experiments/full_model/"
