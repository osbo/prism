#!/bin/bash
#SBATCH --job-name=prism_abl
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-4
# On ORCD Engaging you may need: --partition=sched_mit_gpu
#
# Submit all five ablations in parallel:
#   sbatch slurm_ablations.sh
#
# Or run a single variant (e.g. index 2):
#   sbatch --array=2 slurm_ablations.sh

set -euo pipefail
mkdir -p logs

module load cuda miniforge
source activate prism

cd "$SLURM_SUBMIT_DIR"

# ------------------------------------------------------------------
# Ablation definitions  (index → name + lambda overrides)
# All variants use --reduced_model (latent=64, hidden=128, layers=4)
# and train for 30 epochs.  Evaluate on 50 test objects.
# ------------------------------------------------------------------
declare -A NAMES EXTRA_ARGS

NAMES[0]="ablation_baseline"
EXTRA_ARGS[0]=""

NAMES[1]="ablation_no_photometric"
EXTRA_ARGS[1]="--lambda_render 0.0 --lambda_perceptual 0.0"

NAMES[2]="ablation_no_depth"
EXTRA_ARGS[2]="--lambda_depth 0.0"

NAMES[3]="ablation_no_normal"
EXTRA_ARGS[3]="--lambda_normal 0.0"

NAMES[4]="ablation_no_eikonal"
EXTRA_ARGS[4]="--lambda_eikonal 0.0"

NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
EXTRA="${EXTRA_ARGS[$SLURM_ARRAY_TASK_ID]}"

echo "Array task $SLURM_ARRAY_TASK_ID → experiment: $NAME"
echo "Extra args: $EXTRA"

# shellcheck disable=SC2086  # intentional word-splitting for EXTRA
python run_experiment.py \
    --exp_name "$NAME" \
    --n_epochs 30 \
    --reduced_model \
    --n_eval_objects 50 \
    $EXTRA

echo "Ablation $NAME done.  Results in experiments/$NAME/"
