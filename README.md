# Backdoors Experiment Runner

This project provides tools for running backdoor attack experiments on various vision models.

## Getting Started

To set up the project, run:
```bash
./setup.sh
```
This script will create a virtual environment, install dependencies, and create a `.env` file from `.env.example`. Make sure to update the `.env` file with **WANDB_API_KEY** to use Weights & Biases logging.

## Scripts Overview

### 1. `single.sh`
A wrapper script to run a single experiment. It simplifies calling `src/main.py` by providing sensible defaults and handling virtual environment detection.

**Usage:**
```bash
./single.sh --model-name <model_name> [options]
```

**Common Options:**
- `-mn, --model-name`: (Required) e.g., `resnet152`, `vit_b_16`, `efficientnet_b4`.
- `-m, --model-config`: Model configuration file (default: `default.json`).
- `-d, --dataset`: Dataset configuration (default: `default.json`).
- `-t, --training`: Training configuration (default: `default.json`).
- `-bd, --backdoor`: Backdoor configuration (default: `none`).
- `-obs, --observability`: Observability configuration (default: `default.json`).
- `-g, --gpu`: GPU index to use.
- `--output-path`: Override the default output directory.
- `--archive-results`: Create a zip archive of the run output directory after the run completes, then remove the uncompressed run directory.

**Example:**
```bash
./single.sh -mn resnet152 -bd 1.json -g 0

./single.sh -mn resnet152 -bd 1.json -g 0 --archive-results
```

---

### 2. `batch.py`
A Python script for running multiple experiments defined in a JSON file. It automatically manages GPU queues, running experiments in parallel across different GPUs (one experiment per GPU at a time).

**Usage:**
```bash
python3 batch.py <experiment_json> [options]
```

**Arguments:**
- `experiment_json`: Filename of the experiment specification located in the `experiments/` directory (e.g., `baseline.json`).

**Options:**
- `-n, --dry-run`: Print the commands that would be executed without actually running them.
- `--archive-results`: Archive each run output directory as `<run_dir>.zip` after completion, then remove the uncompressed run directory.

**Example:**
```bash
python3 batch.py baseline.json

python3 batch.py baseline.json --dry-run
```

**Experiment JSON Format**

The JSON file should be a list of groups, where each group specifies a GPU and the parameters for the runs:
```json
[
  {
    "gpu": 0,
    "model_name": "resnet152",
    "backdoors": ["1.json", "2.json", "none"],
    "output": "output/resnet152/",
    "archive_results": true
  },
  {
    "gpu": 1,
    "model_name": "vit_b_16",
    "backdoors": ["1.json"],
    "output": "output/vit_b_16/",
    "archive_results": false
  }
]
```
In the example above, `batch.py` will start two parallel workers (one for GPU 0 and one for GPU 1).

## Configuration
All configuration files are located in the `config/` directory, organized by type:
- `config/models/`: Model architectures and hyper-parameters.
- `config/models/{model_name}/training/`: Model specific hyperparameters.
- `config/datasets/`: Dataset parameters.
- `config/backdoors/`: Trigger definitions and attack parameters.
- `config/observability/`: Settings for monitoring and logging (e.g., image samples).
  - `collect_images_freq`: (int) Frequency (in epochs) of collecting samples. If `0`, monitoring is disabled.
  - `num_images_to_collect`: (int) Number of samples to collect per session.
  Collected artifacts are saved in the run's `output/` directory and logged to Weights & Biases.
