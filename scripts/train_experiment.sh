#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 <experiment-variant> <gpu-id> <scene> <angular-step>" \
        "<dataset-directory> [hydra-overrides...]" >&2
    exit 2
fi

experiment_variant=$1
gpu_id=$2
scene=$3
angular_step=$4
dataset_directory=$5
shift 5
hydra_overrides=("$@")

if (( angular_step < 1 || angular_step > 8 )); then
    echo "angular-step must be between 1 and 8" >&2
    exit 2
fi

angular_grid_size=$((angular_step + 1))
angular_grid_start=$(((9 - angular_grid_size) / 2))
angular_grid_end=$((angular_grid_start + angular_grid_size))

case "$experiment_variant" in
    hylf)
        validation_interval=10
        experiment_name="hylf_${scene}_step_${angular_step}"
        ;;
    ablation_two_plane|ablation_multi_plane|ablation_planar_voxel)
        validation_interval=1
        experiment_name="hylf_${experiment_variant}_${scene}_step_${angular_step}"
        ;;
    *)
        echo "Unknown experiment variant: $experiment_variant" >&2
        exit 2
        ;;
esac

regularizer_override=()
if [[ "$experiment_variant" != "ablation_planar_voxel" ]]; then
    regularizer_override+=(experiment/regularizers/planar_voxel=tv)
fi

python_executable=${PYTHON_BIN:-python}
data_root=${DATA_ROOT:-/data}
max_epochs=${NUM_EPOCHS:-30}
render_interval=${RENDER_EVERY:-20}
print_loss=${PRINT_LOSS:-True}
checkpoint_interval=${CKPT_EVERY:-10}
cpu_threads=${OMP_NUM_THREADS:-1}
if (( cpu_threads < 1 )); then
    cpu_threads=1
fi

OMP_NUM_THREADS="$cpu_threads" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
CUDA_VISIBLE_DEVICES="$gpu_id" "$python_executable" train.py \
    experiment/dataset=sparse_angular \
    experiment/training=hylf \
    experiment/model="$experiment_variant" \
    experiment.training.val_every="$validation_interval" \
    experiment.training.ckpt_every="$checkpoint_interval" \
    experiment.training.test_every="$validation_interval" \
    experiment.training.render_every="$render_interval" \
    experiment.training.num_epochs="$max_epochs" \
    experiment.params.print_loss="$print_loss" \
    experiment.params.name="$experiment_name" \
    experiment.params.data_dir="$data_root" \
    experiment.dataset.scene="$scene" \
    experiment.dataset.dataset_directory="$dataset_directory" \
    experiment.dataset.angular_grid.step="$angular_step" \
    experiment.dataset.angular_grid.start_row="$angular_grid_start" \
    experiment.dataset.angular_grid.end_row="$angular_grid_end" \
    experiment.dataset.angular_grid.start_col="$angular_grid_start" \
    experiment.dataset.angular_grid.end_col="$angular_grid_end" \
    "${regularizer_override[@]}" \
    "${hydra_overrides[@]}"
