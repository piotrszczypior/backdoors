import argparse
import os
from pathlib import Path

from config.ConfigLoader import ConfigLoader, GlobalConfig


def get_args_parser():
    parser = argparse.ArgumentParser(description="Backdoor Training")

    # fmt: off
    parser.add_argument("--model", default="resnet152", type=str, help="model name")

    parser.add_argument("--model-config", default="default.json", type=str, help="model config file name")
    parser.add_argument("--dataset-config", default="default.json", type=str, help="dataset config name")
    parser.add_argument("--training-config", default="default.json", type=str, help="training config file name")
    parser.add_argument("--wandb-config", default="default.json", type=str, help="wandb config name")
    parser.add_argument("--localfs-config", default="default.json", type=str, help="localfs config name")
    parser.add_argument("--config-dir", default="config", type=str, help="config directory")
    parser.add_argument("--backdoor-config", default=None, type=str, help="backdoor config name")

    parser.add_argument("--amp", action="store_true", help="use mixed precision (FP16)")
    # fmt: on

    return parser


def get_config(args: argparse.Namespace) -> GlobalConfig:
    config_dir = Path(args.config_dir)

    assert config_dir.is_dir(), f"Config directory '{args.config_dir}' does not exist"

    model_path = config_dir / "models" / args.model / args.model_config
    assert model_path.is_file(), f"Model config not found at: {model_path}"

    dataset_path = config_dir / "datasets" / args.dataset_config
    assert dataset_path.is_file(), f"Dataset config not found at: {dataset_path}"

    training_path = config_dir / "training" / args.training_config
    assert training_path.is_file(), f"Training config not found at: {training_path}"

    wandb_path = config_dir / "wandb" / args.wandb_config
    assert wandb_path.is_file(), f"Wandb config not found at: {wandb_path}"

    localfs_path = config_dir / "localfs" / args.localfs_config
    assert localfs_path.is_file(), f"LocalFS config not found at: {localfs_path}"

    backdoor_path = None
    if args.backdoor_config:
        backdoor_path = config_dir / "backdoors" / args.backdoor_config / "default.json"
        assert backdoor_path.is_file(), f"Backdoor config not found at: {backdoor_path}"

    config = ConfigLoader.load(
        model_name=args.model,
        model_config_path=model_path,
        dataset_config_path=dataset_path,
        training_config_path=training_path,
        wandb_config_path=wandb_path,
        localfs_config_path=localfs_path,
        backdoor_config_path=backdoor_path,
    )

    return config
