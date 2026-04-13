from __future__ import annotations
import os
import shutil
import zipfile
from pathlib import Path

from .Log import Log
from config.ConfigLoader import GlobalConfig
from output.run_artifacts import get_run_output_dir

log = Log.for_source(__name__)


def cleanup_and_archive_run_artifacts(config: GlobalConfig) -> None:
    run_dir = get_run_output_dir(config)
    log.information("cleanup_started", run_dir=str(run_dir))

    try:
        _archive_logs(run_dir)
        _remove_checkpoints(run_dir)
        _remove_images(run_dir)
        _remove_configs(run_dir)
        log.information("cleanup_finished", run_dir=str(run_dir))
    except Exception as e:
        log.error("cleanup_failed", error=str(e))


def _archive_logs(run_dir: Path) -> None:
    log_files = list(run_dir.glob("*.log"))
    if not log_files:
        log.warning("no_log_files_found")
        return
    
    # FIXME:
    archive_path = run_dir.parent / f"{run_dir.name}.zip"
    # if archive_path.exists():
    #     archive_path.unlink()

    # shutil.make_archive(
    #     base_name=str(archive_path.with_suffix("")),
    #     format="zip",
    #     root_dir=str(run_dir.parent),
    #     base_dir=run_dir.name,
    # )
    # shutil.rmtree(run_dir)
    pass


def _remove_checkpoints(run_dir: Path) -> None:
    for model_path in run_dir.glob("*.pth"):
        if model_path.exists():
            log.information("removing_checkpoint", checkpoint_path=str(model_path))
            os.remove(model_path)


def _remove_images(run_dir: Path) -> None:
    images_dir = run_dir / "images"
    if images_dir.exists() and images_dir.is_dir():
        log.information("removing_images", images_dir=str(images_dir))
        shutil.rmtree(images_dir)


def _remove_configs(run_dir: Path) -> None:
    configs_dir = run_dir / "configs"
    if configs_dir.exists() and configs_dir.is_dir():
        log.information("removing_configs", configs_dir=str(configs_dir))
        shutil.rmtree(configs_dir)
