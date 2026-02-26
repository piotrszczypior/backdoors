#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

usage() {
  cat <<'EOF'
Usage:
  ./batch.sh [options]

Options (value can be a literal config path relative to its root, a glob, or a comma-separated list):
  --models,   -m   (optional) model configs rooted at config/models
                     default: */*.json
  --datasets, -d   (optional) dataset configs rooted at config/datasets
                     default: *.json
  --backdoor, -bd  (optional) backdoor configs rooted at config/backdoors/<chosen-dataset>/
                     default: *.json
                     special value: none (run clean / no backdoor)
  --wandb          (optional) wandb configs rooted at config/wandb
                     default: default.json
  --localfs        (optional) localfs configs rooted at config/localfs
                     default: default.json
  --cuda           (optional) cuda configs rooted at config/cuda
                     default: default.json
  --dry-run, -n    Print generated commands only; do not execute
  --help,     -h   Show help

Examples:
  ./batch.sh
  ./batch.sh -m 'resnet152/*.json,vit_b_16/*.json' -d '*.json'
  ./batch.sh -m resnet152/default.json -d default.json -bd none
  ./batch.sh --backdoor '*.json'
  ./batch.sh --dry-run -m 'resnet152/*.json' -d '*.json'
EOF
}

die() { echo "Error: $*" >&2; exit 1; }
warn() { echo "Warning: $*" >&2; }

need_value() {
  [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; usage; exit 1; }
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SINGLE_SH="$SCRIPT_DIR/single.sh"
CUDA_DEVICE_QUERY_PY="$SCRIPT_DIR/src/query_cuda_devices.py"
[[ -f "$SINGLE_SH" ]] || die "single.sh not found at $SINGLE_SH"
[[ -f "$CUDA_DEVICE_QUERY_PY" ]] || die "CUDA device query script not found at $CUDA_DEVICE_QUERY_PY"

declare -a MODEL_SPECS=() DATASET_SPECS=() BACKDOOR_SPECS=() WANDB_SPECS=() LOCALFS_SPECS=() CUDA_SPECS=()
DRY_RUN=0

append_csv_specs() {
  local array_name=$1 raw=$2
  local -n out=$array_name
  local -a parts=()
  local part

  IFS=',' read -r -a parts <<<"$raw"
  for part in "${parts[@]}"; do
    [[ -n "$part" ]] && out+=("$part")
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models|-m)    need_value "$@"; append_csv_specs MODEL_SPECS "$2"; shift 2 ;;
    --datasets|-d)  need_value "$@"; append_csv_specs DATASET_SPECS "$2"; shift 2 ;;
    --backdoor|-bd) need_value "$@"; append_csv_specs BACKDOOR_SPECS "$2"; shift 2 ;;
    --wandb)        need_value "$@"; append_csv_specs WANDB_SPECS "$2"; shift 2 ;;
    --localfs)      need_value "$@"; append_csv_specs LOCALFS_SPECS "$2"; shift 2 ;;
    --cuda)         need_value "$@"; append_csv_specs CUDA_SPECS "$2"; shift 2 ;;
    --dry-run|-n)   DRY_RUN=1; shift ;;
    --help|-h)      usage; exit 0 ;;
    *)              echo "Error: unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

has_glob() {
  [[ "$1" == *"*"* || "$1" == *"?"* || "$1" == *"["* ]]
}

dedupe_array() {
  local in_name=$1 out_name=$2
  local -n in_arr=$in_name out_arr=$out_name
  local -A seen=()
  local item

  out_arr=()
  for item in "${in_arr[@]}"; do
    [[ -n "${seen[$item]+x}" ]] && continue
    seen["$item"]=1
    out_arr+=("$item")
  done
}

detect_available_gpus() {
  local out_name=$1
  local -n out_ref=$out_name
  local line
  local py_bin
  local lister_output
  local lister_status

  out_ref=()
  py_bin=${PYTHON_BIN:-python}
  command -v "$py_bin" >/dev/null 2>&1 || die "python executable not found: $py_bin"

  lister_output=""
  lister_status=0
  if ! lister_output=$("$py_bin" "$CUDA_DEVICE_QUERY_PY" --ids-only 2>&1); then
    lister_status=$?
    [[ -n "$lister_output" ]] && warn "CUDA device lister failed (exit $lister_status): $lister_output"
  fi

  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*([0-9]+)[[:space:]]*$ ]] || continue
    out_ref+=("${BASH_REMATCH[1]}")
  done <<<"$lister_output"

  local -a tmp=("${out_ref[@]}")
  dedupe_array tmp out_ref

  [[ ${#out_ref[@]} -gt 0 ]] || die "no CUDA devices detected by $CUDA_DEVICE_QUERY_PY"
}

build_job_cmd() {
  local idx=$1 gpu=$2 out_name=$3
  local -n out_ref=$out_name

  out_ref=(
    bash "$SINGLE_SH"
    -m "${JOB_MODEL_SPECS[idx]}"
    -d "${JOB_DATASET_SPECS[idx]}"
    --wandb "${JOB_WANDB_SPECS[idx]}"
    --localfs "${JOB_LOCALFS_SPECS[idx]}"
    --gpu "$gpu"
  )
  [[ "${JOB_BACKDOOR_SPECS[idx]}" != "none" ]] && out_ref+=(-bd "${JOB_BACKDOOR_SPECS[idx]}")
}

resolve_specs() {
  local root=$1 default_spec=$2 input_name=$3 output_name=$4 allow_none=${5:-0}
  local -n input_ref=$input_name output_ref=$output_name
  local -a specs=()
  local spec

  if [[ ${#input_ref[@]} -eq 0 ]]; then
    IFS=',' read -r -a specs <<<"$default_spec"
  else
    specs=("${input_ref[@]}")
  fi

  output_ref=()

  for spec in "${specs[@]}"; do
    [[ -n "$spec" ]] || continue

    if [[ $allow_none -eq 1 && ( "$spec" == "none" || "$spec" == "-" ) ]]; then
      output_ref+=("none")
      continue
    fi

    if has_glob "$spec"; then
      [[ -d "$root" ]] || continue

      local found=0 match
      while IFS= read -r -d '' match; do
        output_ref+=("$match")
        found=1
      done < <(
        cd "$root"
        for match in $spec; do
          [[ -f "$match" ]] && printf '%s\0' "$match"
        done
      )

      if [[ $found -eq 0 && ${#input_ref[@]} -gt 0 ]]; then
        warn "no matches for pattern '$spec' in $root"
      fi
    else
      [[ -f "$root/$spec" ]] || die "file not found: $root/$spec"
      output_ref+=("$spec")
    fi
  done

  local -a tmp=("${output_ref[@]}")
  dedupe_array tmp output_ref
}

declare -a RESOLVED_MODELS=() RESOLVED_DATASETS=() RESOLVED_WANDB=() RESOLVED_LOCALFS=() RESOLVED_CUDA=()
resolve_specs "$SCRIPT_DIR/config/models"   "*/*.json"     MODEL_SPECS   RESOLVED_MODELS
resolve_specs "$SCRIPT_DIR/config/datasets" "*.json"       DATASET_SPECS RESOLVED_DATASETS
resolve_specs "$SCRIPT_DIR/config/wandb"    "default.json" WANDB_SPECS   RESOLVED_WANDB
resolve_specs "$SCRIPT_DIR/config/localfs"  "default.json" LOCALFS_SPECS RESOLVED_LOCALFS
resolve_specs "$SCRIPT_DIR/config/cuda"     "default.json" CUDA_SPECS    RESOLVED_CUDA 1

[[ ${#RESOLVED_MODELS[@]}   -gt 0 ]] || die "no model configs matched under config/models"
[[ ${#RESOLVED_DATASETS[@]} -gt 0 ]] || die "no dataset configs matched under config/datasets"
[[ ${#RESOLVED_WANDB[@]}    -gt 0 ]] || die "no wandb configs matched under config/wandb"
[[ ${#RESOLVED_LOCALFS[@]}  -gt 0 ]] || die "no localfs configs matched under config/localfs"

declare -a JOB_MODEL_SPECS=() JOB_DATASET_SPECS=() JOB_BACKDOOR_SPECS=() JOB_WANDB_SPECS=() JOB_LOCALFS_SPECS=()
run_count=0
pass_count=0
fail_count=0
skip_count=0
declare -a FAILED_COMMANDS=()

for model_spec in "${RESOLVED_MODELS[@]}"; do
  for dataset_spec in "${RESOLVED_DATASETS[@]}"; do
    dataset_root=$(dirname "$dataset_spec")
    [[ "$dataset_root" == "." ]] && dataset_root=""

    backdoor_root="$SCRIPT_DIR/config/backdoors"
    [[ -n "$dataset_root" ]] && backdoor_root="$backdoor_root/$dataset_root"

    declare -a RESOLVED_BACKDOORS=()
    resolve_specs "$backdoor_root" "*.json" BACKDOOR_SPECS RESOLVED_BACKDOORS 1

    if [[ ${#RESOLVED_BACKDOORS[@]} -eq 0 ]]; then
      if [[ ${#BACKDOOR_SPECS[@]} -eq 0 ]]; then
        warn "no backdoor configs matched for dataset '$dataset_spec' under $backdoor_root; skipping dataset/model permutations"
      else
        warn "no backdoor configs matched for dataset '$dataset_spec' under $backdoor_root"
      fi
      ((++skip_count))
      continue
    fi

    for backdoor_spec in "${RESOLVED_BACKDOORS[@]}"; do
      for wandb_spec in "${RESOLVED_WANDB[@]}"; do
        for localfs_spec in "${RESOLVED_LOCALFS[@]}"; do
          JOB_MODEL_SPECS+=("$model_spec")
          JOB_DATASET_SPECS+=("$dataset_spec")
          JOB_BACKDOOR_SPECS+=("$backdoor_spec")
          JOB_WANDB_SPECS+=("$wandb_spec")
          JOB_LOCALFS_SPECS+=("$localfs_spec")
          ((++run_count))
        done
      done
    done
  done
done

[[ $run_count -gt 0 ]] || die "no runs were generated"

declare -a AVAILABLE_GPUS=()
if [[ ${#RESOLVED_CUDA[@]} -gt 0 && "${RESOLVED_CUDA[0]}" != "none" ]]; then
  cuda_config_path="$SCRIPT_DIR/config/cuda/${RESOLVED_CUDA[0]}"
  echo "Loading GPUs from config: $cuda_config_path"
  py_bin=${PYTHON_BIN:-python}
  AVAILABLE_GPUS=($("$py_bin" -c "import json; print(' '.join(map(str, json.load(open('$cuda_config_path')))))"))
else
  detect_available_gpus AVAILABLE_GPUS
fi

gpu_count=${#AVAILABLE_GPUS[@]}
echo "Target GPUs: ${AVAILABLE_GPUS[*]}"

if [[ $DRY_RUN -eq 1 ]]; then
  for ((i=0; i<run_count; i++)); do
    gpu="${AVAILABLE_GPUS[i % gpu_count]}"
    build_job_cmd "$i" "$gpu" cmd
    echo "[$((i + 1))][gpu=$gpu] ${cmd[*]}"
    ((++pass_count))
  done
else
  RESULTS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/batch-gpu-results.XXXXXX")
  trap 'rm -rf "$RESULTS_DIR"' EXIT

  run_worker() {
    local worker_slot=$1 gpu=$2 gpu_stride=$3
    local result_file="$RESULTS_DIR/worker_${worker_slot}.tsv"
    local -a cmd=()
    local status i
    : >"$result_file"

    for ((i=worker_slot; i<run_count; i+=gpu_stride)); do
      build_job_cmd "$i" "$gpu" cmd
      echo "[$((i + 1))][gpu=$gpu] ${cmd[*]}"

      if "${cmd[@]}"; then
        printf 'PASS\t%d\n' "$i" >>"$result_file"
      else
        status=$?
        printf 'FAIL\t%d\t%d\t%s\n' "$i" "$status" "${cmd[*]}" >>"$result_file"
        warn "command failed on gpu $gpu with exit code $status (continuing)"
      fi
    done
  }

  declare -a WORKER_PIDS=()
  worker_count=$(( run_count < gpu_count ? run_count : gpu_count ))
  for ((slot=0; slot<worker_count; slot++)); do
    gpu="${AVAILABLE_GPUS[slot]}"
    run_worker "$slot" "$gpu" "$gpu_count" &
    WORKER_PIDS+=("$!")
  done

  worker_failure=0
  for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "$pid"; then
      worker_failure=1
    fi
  done
  [[ $worker_failure -eq 0 ]] || die "one or more GPU worker processes crashed"

  while IFS= read -r -d '' result_file; do
    while IFS=$'\t' read -r kind _idx status cmd_text; do
      [[ -n "$kind" ]] || continue
      if [[ "$kind" == "PASS" ]]; then
        ((++pass_count))
      elif [[ "$kind" == "FAIL" ]]; then
        ((++fail_count))
        FAILED_COMMANDS+=("exit=$status :: $cmd_text")
      fi
    done <"$result_file"
  done < <(find "$RESULTS_DIR" -type f -name 'worker_*.tsv' -print0)
fi

echo
echo "Batch summary:"
echo "  total generated : $run_count"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "  dry-run         : yes (no commands executed)"
  echo "  printable cmds  : $pass_count"
else
  echo "  passed          : $pass_count"
  echo "  failed          : $fail_count"
fi
echo "  skipped groups  : $skip_count"

if [[ $DRY_RUN -eq 0 && $fail_count -gt 0 ]]; then
  echo
  echo "Failed commands:"
  i=0
  for entry in "${FAILED_COMMANDS[@]}"; do
    ((++i))
    echo "  [$i] $entry"
  done
  exit 1
fi
