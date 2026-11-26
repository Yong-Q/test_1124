#!/bin/bash
#SBATCH --job-name=xi_github_examples
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
set -euo pipefail
source xi_cnn_remaining_trial/environment.sh
base="$PWD/xi_sognet_atomic"
main="$base/xi_readout_1000"
trial="$main/short_spatial"
export PYTHONPATH="$main/grid3d_supervision/full_frames/external_c1_calibration:$main/grid3d_supervision/full_frames/real_grid/pooled_reference/regional_batch:$main/grid3d_supervision/full_frames/real_grid/pooled_reference:$main/grid3d_supervision/full_frames/real_grid:$main/grid3d_supervision:$main/grid_short_spatial:$trial:$main/angular_spatial:$main/spatial_xi:$main/xi_only_coupling:$main/conditional_radial:$main:$base:$base/loss_trial:$base/curvature_mu:$PWD/xi_cnn_remaining_trial:$PWD/xi_3d_charge_20261117_v1:$PYTHONPATH"
folder="$main/grid3d_supervision/full_frames/mixed_regional_grid"
export PYTHONPATH="$folder/c1_bulk_c2/c1_reference:$folder/c1_bulk_c2:$folder:$PYTHONPATH"
"$python" -u "$PWD/test_1124_delivery/fex_models/provenance/export_native_examples.py" "$PWD/test_1124_delivery/fex_models/examples" "$@"
