from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from output.Log import Log
from output.run_artifacts import get_run_output_dir

if TYPE_CHECKING:
    from config.ConfigLoader import GlobalConfig


class _Checkpoint:
    def __init__(self):
        self._base_dir: Path = Path(".")
        self._is_initialized: bool = False
        self._log = Log.for_source(__name__)

    def initialize(self, config: "GlobalConfig"):
        checkpoint_dir = get_run_output_dir(config)

        self._base_dir = checkpoint_dir
        self._is_initialized = True
        self._log.information(
            "checkpoint_output_configured", checkpoint_dir=str(checkpoint_dir)
        )

    def path(self, filename: str) -> Path:
        if not self._is_initialized:
            self._log.warning(
                "checkpoint_not_initialized_using_default_dir",
                default_dir=str(self._base_dir),
            )
        path = self._base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, filename: str, obj: Any) -> Path:
        path = self.path(filename)
        torch.save(obj, path)
        self._log.information("checkpoint_saved", path=str(path))
        return path

    def save_model(self, obj: Any) -> Path:
        return self.save("model.pth", obj)


Checkpoint = _Checkpoint()
