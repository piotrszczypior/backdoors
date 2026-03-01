from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from config.ConfigLoader import GlobalConfig


def get_run_output_dir(config: GlobalConfig) -> Path:
    """Returns the directory where run artifacts should be written."""

    if config.output_path:
        run_dir = Path(config.output_path)
    else:
        base_dir = Path(
            config.localfs_config.output_dir if config.localfs_config else "."
        )
        backdoor_name = (
            config.backdoor_config.name if config.backdoor_config else "clean"
        )
        run_dir = (
            base_dir
            / config.dataset_config.name
            / backdoor_name
            / config.model_config.name
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def dump_config_artifacts(config: GlobalConfig) -> Path:
    """Copies the JSON configs that defined this run into the output directory."""

    run_dir = get_run_output_dir(config)
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    _copy_config(config.model_config, configs_dir / "model.json")
    _copy_config(config.training_config, configs_dir / "training.json")
    _copy_config(config.dataset_config, configs_dir / "dataset.json")
    _copy_config(config.wandb_config, configs_dir / "wandb.json")

    if config.backdoor_config:
        _copy_config(config.backdoor_config, configs_dir / "backdoor.json")
    else:
        _write_clean_backdoor_marker(configs_dir / "backdoor.json")

    return configs_dir


def _copy_config(config_obj: Optional[object], destination: Path) -> None:
    json_path = getattr(config_obj, "json_path", None)
    if not json_path:
        return

    shutil.copy2(json_path, destination)


def _write_clean_backdoor_marker(path: Path) -> None:
    payload = {"enabled": False, "name": "none"}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
