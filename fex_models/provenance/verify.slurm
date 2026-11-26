#!/bin/bash
#SBATCH --job-name=xi_repo_verify
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail
source xi_cnn_remaining_trial/environment.sh
cd test_1124_delivery
"$python" -u -m fex_models.verify "runs/fex_verify_${SLURM_JOB_ID}" --device cpu --device cuda
