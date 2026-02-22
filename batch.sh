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
[[ -f "$SINGLE_SH" ]] || die "single.sh not found at $SINGLE_SH"

declare -a MODEL_SPECS=() DATASET_SPECS=() BACKDOOR_SPECS=() WANDB_SPECS=() LOCALFS_SPECS=()
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
  local -n in_ref=$in_name out_ref=$out_name
  local -A seen=()
  local item

  out_ref=()
  for item in "${in_ref[@]}"; do
    [[ -n "${seen[$item]+x}" ]] && continue
    seen["$item"]=1
    out_ref+=("$item")
  done
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

declare -a RESOLVED_MODELS=() RESOLVED_DATASETS=() RESOLVED_WANDB=() RESOLVED_LOCALFS=()
resolve_specs "$SCRIPT_DIR/config/models"   "*/*.json"     MODEL_SPECS   RESOLVED_MODELS
resolve_specs "$SCRIPT_DIR/config/datasets" "*.json"       DATASET_SPECS RESOLVED_DATASETS
resolve_specs "$SCRIPT_DIR/config/wandb"    "default.json" WANDB_SPECS   RESOLVED_WANDB
resolve_specs "$SCRIPT_DIR/config/localfs"  "default.json" LOCALFS_SPECS RESOLVED_LOCALFS

[[ ${#RESOLVED_MODELS[@]}   -gt 0 ]] || die "no model configs matched under config/models"
[[ ${#RESOLVED_DATASETS[@]} -gt 0 ]] || die "no dataset configs matched under config/datasets"
[[ ${#RESOLVED_WANDB[@]}    -gt 0 ]] || die "no wandb configs matched under config/wandb"
[[ ${#RESOLVED_LOCALFS[@]}  -gt 0 ]] || die "no localfs configs matched under config/localfs"

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
          cmd=(bash "$SINGLE_SH" -m "$model_spec" -d "$dataset_spec" --wandb "$wandb_spec" --localfs "$localfs_spec")
          [[ "$backdoor_spec" != "none" ]] && cmd+=(-bd "$backdoor_spec")

          ((++run_count))
          echo "[$run_count] ${cmd[*]}"

          if [[ $DRY_RUN -eq 1 ]]; then
            ((++pass_count))
            continue
          fi

          if "${cmd[@]}"; then
            ((++pass_count))
          else
            status=$?
            ((++fail_count))
            FAILED_COMMANDS+=("exit=$status :: ${cmd[*]}")
            warn "command failed with exit code $status (continuing)"
          fi
        done
      done
    done
  done
done

[[ $run_count -gt 0 ]] || die "no runs were generated"

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