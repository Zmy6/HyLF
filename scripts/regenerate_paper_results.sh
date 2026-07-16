#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

artifact_root=${ARTIFACT_ROOT:-/root/autodl-tmp/HyLF_runs}
data_root=${DATA_ROOT:-/root/autodl-tmp/InstantSplat-main/data}
python_executable=${PYTHON_BIN:-/root/autodl-tmp/hylf-env/bin/python}
gpu_id=${GPU_ID:-0}
angular_step=${ANGULAR_STEP:-6}
validation_epoch=${VALIDATION_EPOCH:-30}
num_workers=${NUM_WORKERS:-4}
max_parallel=${MAX_PARALLEL:-4}
scene_filter=${SCENES:-}
log_root="$artifact_root/logs"
checkpoint_root="$artifact_root/checkpoints"
run_root="$artifact_root/regenerate_2x2_to_7x7"
metrics_root="$artifact_root/metrics/epoch_$validation_epoch"
status_file="$run_root/status.txt"

scene_specs=(
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

mkdir -p "$log_root" "$run_root/scenes" "$metrics_root"
: > "$status_file"

record_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$status_file"
}

is_selected() {
    local scene=$1
    [[ -z "$scene_filter" || " $scene_filter " == *" $scene "* ]]
}

evaluate_scene() {
    local dataset_directory=$1
    local scene=$2
    local checkpoint="$checkpoint_root/hylf_${scene}_step_${angular_step}/last.ckpt"

    record_status "evaluation-start $dataset_directory/$scene"
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHON_BIN="$python_executable" \
    NUM_EPOCHS="$validation_epoch" \
    RENDER_EVERY=1000 \
    PRINT_LOSS=False \
    bash scripts/train_hylf.sh \
        "$gpu_id" "$scene" "$angular_step" "$dataset_directory" \
        experiment.params.test_only=True \
        experiment.params.evaluation_epoch="$validation_epoch" \
        experiment.params.data_dir="$data_root" \
        experiment.params.log_dir="$log_root" \
        experiment.params.ckpt_dir="$checkpoint_root" \
        experiment.training.num_workers="$num_workers" \
        > "$run_root/scenes/${dataset_directory}_${scene}.log" 2>&1
    record_status "evaluation-complete $dataset_directory/$scene checkpoint=$checkpoint"
}

declare -A active_jobs=()
available_scenes=()
missing_scenes=()
failed_count=0

wait_for_job() {
    local completed_pid=""
    local exit_status=0
    local scene_label
    if wait -n -p completed_pid "${!active_jobs[@]}"; then
        exit_status=0
    else
        exit_status=$?
    fi
    scene_label=${active_jobs[$completed_pid]:-unknown}
    unset 'active_jobs[$completed_pid]'
    if (( exit_status != 0 )); then
        failed_count=$((failed_count + 1))
        record_status "evaluation-failed $scene_label exit_status=$exit_status"
    fi
}

for scene_spec in "${scene_specs[@]}"; do
    dataset_directory=${scene_spec%%:*}
    scene=${scene_spec#*:}
    if ! is_selected "$scene"; then
        continue
    fi

    checkpoint="$checkpoint_root/hylf_${scene}_step_${angular_step}/last.ckpt"
    if [[ ! -f "$checkpoint" ]]; then
        missing_scenes+=("$scene")
        record_status "checkpoint-missing $dataset_directory/$scene path=$checkpoint"
        continue
    fi
    available_scenes+=("$scene")

    while (( ${#active_jobs[@]} >= max_parallel )); do
        wait_for_job
    done
    evaluate_scene "$dataset_directory" "$scene" &
    active_jobs[$!]="$dataset_directory/$scene"
done

while (( ${#active_jobs[@]} > 0 )); do
    wait_for_job
done

if (( failed_count > 0 )); then
    record_status "regeneration-failed jobs=$failed_count"
    exit 1
fi
if (( ${#available_scenes[@]} == 0 )); then
    record_status "regeneration-failed no-checkpoints"
    exit 1
fi

record_status "testY-start scenes=${available_scenes[*]}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python_executable" testY.py \
    --log-root "$log_root" \
    --scenes "${available_scenes[@]}" \
    --angular-step "$angular_step" \
    --validation-epoch "$validation_epoch" \
    --output-directory "$metrics_root" \
    2>&1 | tee "$run_root/testY.log"
record_status "testY-complete output=$metrics_root"

if (( ${#missing_scenes[@]} > 0 )); then
    record_status "regeneration-partial missing=${missing_scenes[*]}"
    exit 3
fi
record_status "regeneration-complete"
