#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./single.sh --model <path> --dataset <path> [--backdoor <path>] [--wandb <file>] [--localfs <file>] [--gpu <id>]
  ./single.sh -m <path> -d <path> [-bd <path>] [-g <id>]

Arguments:
  --model,   -m   (required) model configuration rooted at config/models
                   example: resnet152/default.json
  --dataset, -d   (required) dataset configuration rooted at config/datasets
                   example: default.json
  --backdoor,-bd  (optional) backdoor configuration rooted at config/backdoors
                   example: default.json
  --wandb         (optional) wandb configuration rooted at config/wandb
                   default: default.json
  --localfs       (optional) localfs configuration rooted at config/localfs
                   default: default.json
  --gpu,     -g   (optional) GPU index to use
                   passed to Python as --gpu and used as cuda:<id> in training
  --help,   -h    Show this help

Example:
  ./single.sh \
    --model resnet152/default.json \
    --dataset default.json \
    --backdoor default.json \
    --wandb default.json \
    --localfs default.json \
    --gpu 0
EOF
}

die() { echo "Error: $*" >&2; exit 1; }

need_value() {
  [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; usage; exit 1; }
}

require_file() {
  local path=$1 label=${2:-file}
  [[ -f "$path" ]] || die "$label not found: $path"
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODEL_SPEC=""
DATASET_SPEC=""
BACKDOOR_SPEC=""
WANDB_SPEC="default.json"
LOCALFS_SPEC="default.json"
GPU_INDEX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|-m)    need_value "$@"; MODEL_SPEC=$2; shift 2 ;;
    --dataset|-d)  need_value "$@"; DATASET_SPEC=$2; shift 2 ;;
    --backdoor|-bd) need_value "$@"; BACKDOOR_SPEC=$2; shift 2 ;;
    --wandb)       need_value "$@"; WANDB_SPEC=$2; shift 2 ;;
    --localfs)     need_value "$@"; LOCALFS_SPEC=$2; shift 2 ;;
    --gpu|-g)      need_value "$@"; GPU_INDEX=$2; shift 2 ;;
    --help|-h)     usage; exit 0 ;;
    *)             echo "Error: unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$MODEL_SPEC" && -n "$DATASET_SPEC" ]] || {
  echo "Error: --model and --dataset are required" >&2
  usage
  exit 1
}

MODEL_CONFIG_PATH="$SCRIPT_DIR/config/models/$MODEL_SPEC"
DATASET_CONFIG_PATH="$SCRIPT_DIR/config/datasets/$DATASET_SPEC"
WANDB_CONFIG_PATH="$SCRIPT_DIR/config/wandb/$WANDB_SPEC"
LOCALFS_CONFIG_PATH="$SCRIPT_DIR/config/localfs/$LOCALFS_SPEC"
DEFAULT_TRAINING_CONFIG="$SCRIPT_DIR/config/training/default.json"

require_file "$MODEL_CONFIG_PATH"
require_file "$DATASET_CONFIG_PATH"
require_file "$WANDB_CONFIG_PATH"
require_file "$LOCALFS_CONFIG_PATH"
require_file "$DEFAULT_TRAINING_CONFIG" "default training config"

BACKDOOR_ARGS=()
if [[ -n "$BACKDOOR_SPEC" && "$BACKDOOR_SPEC" != "none" && "$BACKDOOR_SPEC" != "-" ]]; then
  BACKDOOR_CONFIG_PATH="$SCRIPT_DIR/config/backdoors/$BACKDOOR_SPEC"
  require_file "$BACKDOOR_CONFIG_PATH" "backdoor file"
  BACKDOOR_ARGS=(--backdoor-config-path "$BACKDOOR_CONFIG_PATH")
fi

PYTHON_BIN=${PYTHON_BIN:-python}

GPU_ARGS=()
if [[ -n "$GPU_INDEX" ]]; then
  [[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || die "--gpu must be a non-negative integer"
  GPU_ARGS=(--gpu "$GPU_INDEX")
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/src/main.py" \
  --model-config-path "$MODEL_CONFIG_PATH" \
  --dataset-config-path "$DATASET_CONFIG_PATH" \
  --training-config-path "$DEFAULT_TRAINING_CONFIG" \
  --wandb-config-path "$WANDB_CONFIG_PATH" \
  --localfs-config-path "$LOCALFS_CONFIG_PATH" \
  "${BACKDOOR_ARGS[@]}" \
  "${GPU_ARGS[@]}"
