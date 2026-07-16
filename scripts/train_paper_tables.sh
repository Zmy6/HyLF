#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

data_root=${DATA_ROOT:-/root/autodl-tmp/InstantSplat-main/data}
python_executable=${PYTHON_BIN:-/root/autodl-tmp/hylf-env/bin/python}
gpu_id=${GPU_ID:-0}
angular_step=${ANGULAR_STEP:-6}
num_workers=${NUM_WORKERS:-8}
max_epochs=${NUM_EPOCHS:-30}
artifact_root=${ARTIFACT_ROOT:-/root/autodl-tmp/HyLF_runs}
log_root="$artifact_root/logs"
checkpoint_root="$artifact_root/checkpoints"
run_dir=${RUN_DIR:-$artifact_root/queue_2x2_to_7x7}
scene_filter=${SCENES:-}
max_parallel_scenes=${MAX_PARALLEL:-4}
metrics_root=${METRICS_ROOT:-$repo_root/metrics}

paper_scene_specs=(
    HCI_new:bedroom
    HCI_new:bicycle
    HCI_new:herbs
    HCI_new:boxes
    HCI_new:cotton
    HCI_new:dino
    HCI_new:origami
    HCI_new:sideboard
    INRIA:Bee
    INRIA:Buildings
    INRIA:Hublais
    INRIA:MessyDesk
    INRIA:Sculpture
    30scene:Cars
    30scene:Flower1
    30scene:Flower2
    30scene:Rock
    30scene:Seahorse
)

if (( angular_step < 1 || angular_step > 8 )); then
    echo "ANGULAR_STEP must be between 1 and 8" >&2
    exit 2
fi
if (( max_parallel_scenes < 1 )); then
    echo "MAX_PARALLEL must be at least 1" >&2
    exit 2
fi
if [[ ! -x "$python_executable" ]]; then
    echo "Python executable is not available: $python_executable" >&2
    exit 2
fi

mkdir -p "$run_dir/scenes" "$log_root" "$checkpoint_root" "$metrics_root"
status_file="$run_dir/status.txt"
expected_view_count=$(((angular_step + 1) * (angular_step + 1) - 4))

record_queue_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$status_file"
}

is_scene_selected() {
    local scene=$1
    [[ -z "$scene_filter" || " $scene_filter " == *" $scene "* ]]
}

count_completed_views() {
    local scene=$1
    local validation_directory
    validation_directory="$log_root/hylf_${scene}_step_${angular_step}/val_images/$max_epochs"
    if [[ ! -d "$validation_directory" ]]; then
        echo 0
        return
    fi
    find "$validation_directory" -maxdepth 1 -name '*_pred.png' | wc -l
}

record_queue_status \
    "queue-start data_root=$data_root gpu=$gpu_id angular_step=$angular_step"

train_scene() {
    local dataset_directory=$1
    local scene=$2

    record_queue_status "train-start $dataset_directory/$scene"
    PYTHON_BIN="$python_executable" \
    NUM_EPOCHS="$max_epochs" \
    RENDER_EVERY=1000 \
    PRINT_LOSS=False \
    CKPT_EVERY=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    bash scripts/train_hylf.sh \
        "$gpu_id" "$scene" "$angular_step" "$dataset_directory" \
        experiment.params.data_dir="$data_root" \
        experiment.params.log_dir="$log_root" \
        experiment.params.ckpt_dir="$checkpoint_root" \
        experiment.training.num_workers="$num_workers" \
        > "$run_dir/scenes/${dataset_directory}_${scene}.log" 2>&1
    record_queue_status "train-complete $dataset_directory/$scene"
}

declare -A active_scene_jobs=()
failed_scene_count=0

wait_for_scene_job() {
    local completed_pid=""
    local exit_status=0
    local scene_label

    if wait -n -p completed_pid "${!active_scene_jobs[@]}"; then
        exit_status=0
    else
        exit_status=$?
    fi
    scene_label=${active_scene_jobs[$completed_pid]:-unknown}
    unset 'active_scene_jobs[$completed_pid]'
    if (( exit_status != 0 )); then
        failed_scene_count=$((failed_scene_count + 1))
        record_queue_status \
            "train-failed $scene_label exit_status=$exit_status"
    fi
}

for scene_spec in "${paper_scene_specs[@]}"; do
    dataset_directory=${scene_spec%%:*}
    scene=${scene_spec#*:}
    if ! is_scene_selected "$scene"; then
        continue
    fi

    completed_view_count=$(count_completed_views "$scene")
    if (( completed_view_count >= expected_view_count )); then
        record_queue_status \
            "skip-complete $dataset_directory/$scene views=$completed_view_count"
        continue
    fi

    while (( ${#active_scene_jobs[@]} >= max_parallel_scenes )); do
        wait_for_scene_job
    done

    train_scene "$dataset_directory" "$scene" &
    active_scene_jobs[$!]="$dataset_directory/$scene"
done

while (( ${#active_scene_jobs[@]} > 0 )); do
    wait_for_scene_job
done

if (( failed_scene_count > 0 )); then
    record_queue_status "queue-failed scenes=$failed_scene_count"
    exit 1
fi

evaluation_selection=(--dataset all)
evaluation_output_directory=$metrics_root
if [[ -n "$scene_filter" ]]; then
    read -r -a selected_scenes <<< "$scene_filter"
    evaluation_selection=(--scenes "${selected_scenes[@]}")
    evaluation_output_directory="$run_dir/metrics"
    mkdir -p "$evaluation_output_directory"
fi

record_queue_status "evaluation-start"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python_executable" testY.py \
    --log-root "$log_root" \
    "${evaluation_selection[@]}" \
    --angular-step "$angular_step" \
    --validation-epoch "$max_epochs" \
    --output-directory "$evaluation_output_directory" \
    2>&1 | tee "$run_dir/evaluation.log"
record_queue_status "queue-complete"
