#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./single.sh [options]

Arguments:
  --model-name, -mn  (required) model name (e.g. resnet152)
  --model-config, -m (optional) model config file (default: default.json)
  --dataset, -d      (optional) dataset config (default: default.json)
  --training, -t     (optional) training config (default: default.json)
  --backdoor, -bd    (optional) backdoor config (default: none)
  --observability, -obs (optional) observability config (default: default.json)
  --wandb            (optional) wandb config (default: default.json)
  --localfs          (optional) localfs config (default: default.json)
  --output-path      (optional) override run output directory
  --archive-results  (optional) remove checkpoints/images, then zip the run output directory
  --omit-logs        (optional) do not persist local log files
  --omit-models      (optional) do not persist local checkpoints
  --gpu, -g          (optional) GPU index
  --help, -h         Show help
EOF
}

die() { echo "Error: $*" >&2; exit 1; }

need_value() {
  [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; usage; exit 1; }
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODEL_NAME=""
MODEL_CONFIG="default.json"
DATASET_SPEC="default.json"
TRAINING_SPEC="default.json"
BACKDOOR_SPEC=""
OBSERVABILITY_SPEC="default.json"
WANDB_SPEC="default.json"
LOCALFS_SPEC="default.json"
OUTPUT_PATH=""
ARCHIVE_RESULTS=0
OMIT_LOGS=0
OMIT_MODELS=0
GPU_INDEX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name|-mn)   need_value "$@"; MODEL_NAME=$2; shift 2 ;;
    --model-config|-m)  need_value "$@"; MODEL_CONFIG=$2; shift 2 ;;
    --dataset|-d)       need_value "$@"; DATASET_SPEC=$2; shift 2 ;;
    --training|-t)      need_value "$@"; TRAINING_SPEC=$2; shift 2 ;;
    --backdoor|-bd)     need_value "$@"; BACKDOOR_SPEC=$2; shift 2 ;;
    --observability|-obs) need_value "$@"; OBSERVABILITY_SPEC=$2; shift 2 ;;
    --wandb)            need_value "$@"; WANDB_SPEC=$2; shift 2 ;;
    --localfs)          need_value "$@"; LOCALFS_SPEC=$2; shift 2 ;;
    --output-path)      need_value "$@"; OUTPUT_PATH=$2; shift 2 ;;
    --archive-results)  ARCHIVE_RESULTS=1; shift ;;
    --omit-logs)        OMIT_LOGS=1; shift ;;
    --omit-models)      OMIT_MODELS=1; shift ;;
    --gpu|-g)           need_value "$@"; GPU_INDEX=$2; shift 2 ;;
    --help|-h)          usage; exit 0 ;;
    *)                  echo "Error: unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$MODEL_NAME" ]] || die "--model-name is required"

if [[ -f "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON_BIN=${PYTHON_BIN:-python}
fi

BACKDOOR_ARGS=()
if [[ -n "$BACKDOOR_SPEC" && "$BACKDOOR_SPEC" != "none" ]]; then
    BACKDOOR_ARGS=(--backdoor-config "$BACKDOOR_SPEC")
fi

GPU_ARGS=()
if [[ -n "$GPU_INDEX" ]]; then
    GPU_ARGS=(--gpu "$GPU_INDEX")
fi

OUTPUT_ARGS=()
if [[ -n "$OUTPUT_PATH" ]]; then
    OUTPUT_ARGS=(--output-path "$OUTPUT_PATH")
fi

ARCHIVE_ARGS=()
if [[ "$ARCHIVE_RESULTS" -eq 1 ]]; then
    ARCHIVE_ARGS=(--archive-results)
fi

OMIT_LOGS_ARGS=()
if [[ "$OMIT_LOGS" -eq 1 ]]; then
    OMIT_LOGS_ARGS=(--omit-logs)
fi

OMIT_MODELS_ARGS=()
if [[ "$OMIT_MODELS" -eq 1 ]]; then
    OMIT_MODELS_ARGS=(--omit-models)
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/src/main.py" \
  --config-dir "config/" \
  --model-name "$MODEL_NAME" \
  --model-config "$MODEL_CONFIG" \
  --training-config "$TRAINING_SPEC" \
  --dataset-config "$DATASET_SPEC" \
  --wandb-config "$WANDB_SPEC" \
  --observability-config "$OBSERVABILITY_SPEC" \
  "${BACKDOOR_ARGS[@]}" \
  "${GPU_ARGS[@]}" \
  "${ARCHIVE_ARGS[@]}" \
  "${OMIT_LOGS_ARGS[@]}" \
  "${OMIT_MODELS_ARGS[@]}" \
  "${OUTPUT_ARGS[@]}"
