#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./batch_2.sh <experiment_json> [options]

Arguments:
  experiment_json   Path to JSON file in experiments/ (e.g. baseline.json)

Options:
  --dry-run, -n     Print commands only
  --help, -h        Show help
EOF
}

die() { echo "Error: $*" >&2; exit 1; }
warn() { echo "Warning: $*" >&2; }

[[ $# -ge 1 ]] || { usage; exit 1; }

EXP_SPEC=$1
shift
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n) DRY_RUN=1; shift ;;
    --help|-h)    usage; exit 0 ;;
    *)            die "Unknown argument: $1" ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SINGLE_SH="$SCRIPT_DIR/single.sh"
EXP_PATH="$SCRIPT_DIR/experiments/$EXP_SPEC"

[[ -f "$EXP_PATH" ]] || die "Experiment config not found: $EXP_PATH"

if [[ -f "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON_BIN=${PYTHON_BIN:-python}
fi
RESOLVED_JOBS=$( "$PYTHON_BIN" - "$EXP_PATH" <<'EOF'
import json, os, sys
exp_path = sys.argv[1]
with open(exp_path) as f:
    config = json.load(f)

for group in config:
    gpu = group.get('gpu', 0)
    model_name = group['model_name']
    model_config = group.get('model', 'default.json')
    dataset = group.get('dataset', 'default.json')
    training = group.get('training', 'default.json')
    backdoors = group.get('backdoors', ['none'])
    output_base = group.get('output', 'output')
    
    for i, bd in enumerate(backdoors):
        run_output = os.path.join(output_base, f"run{i}")
        print(f"{gpu}\t{model_name}\t{model_config}\t{dataset}\t{training}\t{bd}\t{run_output}")
EOF
)

declare -A GPU_JOBS
GPUS=()

while IFS=$'\t' read -r gpu model_name model_config dataset training backdoor output; do
    [[ -n "$gpu" ]] || continue
    if [[ -z "${GPU_JOBS[$gpu]+x}" ]]; then
        GPUS+=("$gpu")
        GPU_JOBS[$gpu]=""
    fi
    
    GPU_JOBS[$gpu]+="$model_name|$model_config|$dataset|$training|$backdoor|$output"$'\n'
done <<< "$RESOLVED_JOBS"

echo "Loaded experiment: $EXP_SPEC"
echo "Target GPUs: ${GPUS[*]}"
echo

if [[ $DRY_RUN -eq 1 ]]; then
    for gpu in "${GPUS[@]}"; do
        echo "=== GPU $gpu ==="
        while IFS='|' read -r mn mc ds tr bd out; do
            [[ -n "$mn" ]] || continue
            echo "  ./single.sh -mn $mn -m $mc -d $ds -t $tr -bd $bd -g $gpu --localfs $out"
        done <<< "${GPU_JOBS[$gpu]}"
    done
    exit 0
fi

RESULTS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/batch2-results.XXXXXX")
trap 'rm -rf "$RESULTS_DIR"' EXIT

run_gpu_worker() {
    local gpu=$1
    local jobs=$2
    local result_file="$RESULTS_DIR/gpu_${gpu}.log"
    : > "$result_file"
    
    while IFS='|' read -r mn mc ds tr bd out; do
        [[ -n "$mn" ]] || continue
        
        echo "[GPU $gpu] Starting: $mn | $bd"
        if bash "$SINGLE_SH" -mn "$mn" -m "$mc" -d "$ds" -t "$tr" -bd "$bd" -g "$gpu" --localfs "$out"; then
            echo "PASS|$mn|$bd" >> "$result_file"
        else
            echo "FAIL|$mn|$bd" >> "$result_file"
            warn "Job failed on GPU $gpu: $mn | $bd"
        fi
    done <<< "$jobs"
}

declare -a WORKER_PIDS=()
for gpu in "${GPUS[@]}"; do
    run_gpu_worker "$gpu" "${GPU_JOBS[$gpu]}" &
    WORKER_PIDS+=("$!")
done

for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" || warn "A worker process exited with error"
done

pass_total=0
fail_total=0

for gpu in "${GPUS[@]}"; do
    res_file="$RESULTS_DIR/gpu_${gpu}.log"
    [[ -f "$res_file" ]] || continue
    while IFS='|' read -r status m bd; do
        if [[ "$status" == "PASS" ]]; then
            ((++pass_total))
        else
            ((++fail_total))
        fi
    done < "$res_file"
done

echo
echo "Batch Summary:"
echo "  Passed: $pass_total"
echo "  Failed: $fail_total"

[[ $fail_total -eq 0 ]] || exit 1
