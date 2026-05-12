from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from config.ConfigLoader import GlobalConfig
from output.run_artifacts import get_run_output_dir

from .Log import Log

log = Log.for_source(__name__)


def cleanup_and_archive_run_artifacts(config: GlobalConfig) -> Path:
    run_dir = get_run_output_dir(config)
    archive_path = run_dir.parent / f"{run_dir.name}.zip"

    log.information(
        "cleanup_started", run_dir=str(run_dir), archive_path=str(archive_path)
    )

    _remove_checkpoints(run_dir)
    _remove_images(run_dir)

    log.information("archiving_run_artifacts", archive_path=str(archive_path))
    logging.shutdown()
    _archive_run_dir(run_dir, archive_path)

    return archive_path


def _archive_run_dir(run_dir: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()

    shutil.make_archive(
        base_name=str(archive_path.with_suffix("")),
        format="zip",
        root_dir=str(run_dir.parent),
        base_dir=run_dir.name,
    )

    shutil.rmtree(run_dir)


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
