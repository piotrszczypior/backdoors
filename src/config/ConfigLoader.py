from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import importlib
import pkgutil

from config.abstract.AbstractConfig import AbstractConfig
from config.ConfigFactory import ConfigFactory


def _discover_configs():
    config_root = Path(__file__).parent
    package = __package__ + "." if __package__ is not None else ""

    for _, name, is_pkg in pkgutil.walk_packages([str(config_root)], package):
        if not is_pkg and name.endswith("Config"):
            try:
                importlib.import_module(name)
            except (ImportError, TypeError):
                pass


_discover_configs()


# FIXME: types?
@dataclass(frozen=True)
class GlobalConfig:
    model_config: AbstractConfig
    dataset_config: AbstractConfig
    backdoor_config: Optional[AbstractConfig] = None
    training_config: Optional[AbstractConfig] = None
    wandb_config: Optional[AbstractConfig] = None
    localfs_config: Optional[AbstractConfig] = None
    device: Optional[str] = None


class ConfigLoader:
    @staticmethod
    def load(
        model_name: str,
        model_config_path: str,
        dataset_config_path: str,
        training_config_path: str,
        wandb_config_path: str,
        localfs_config_path: str,
        backdoor_config_path: str,
        device: Optional[str] = None,
    ) -> GlobalConfig:
        model_config = ConfigFactory.load("model", model_config_path, model_name)
        dataset_config = ConfigFactory.load("dataset", dataset_config_path)
        training_config = ConfigFactory.load("training", training_config_path)
        wandb_config = ConfigFactory.load("wandb", wandb_config_path)
        localfs_config = ConfigFactory.load("localfs", localfs_config_path)

        backdoor_config = None
        if backdoor_config_path:
            backdoor_config = ConfigFactory.load("backdoor", backdoor_config_path)

        return GlobalConfig(
            model_config=model_config,
            dataset_config=dataset_config,
            training_config=training_config,
            wandb_config=wandb_config,
            localfs_config=localfs_config,
            backdoor_config=backdoor_config,
            device=device,
        )
