import argparse
from pathlib import Path

from config.ConfigLoader import ConfigLoader, GlobalConfig


def get_args_parser():
    parser = argparse.ArgumentParser(description="Backdoor Training")

    # fmt: off
    parser.add_argument("--config-dir", default="config", type=str, help="config directory")

    parser.add_argument("--model-name", default="resnet152", type=str, help="model name")

    parser.add_argument("--model-config", default="default.json", type=str, help="model config file name")
    parser.add_argument("--training-config", default="default.json", type=str, help="training config file name")
    parser.add_argument("--dataset-config", default="default.json", type=str, help="dataset config name")
    parser.add_argument("--wandb-config", default="default.json", type=str, help="wandb config name")
    parser.add_argument("--backdoor-config", default=None, type=str, help="backdoor config name")
    parser.add_argument("--output-path", default="default.json", type=str, help="Output path")
    parser.add_argument("--gpu", default=None, type=int, help="GPU index to use (e.g. 0 -> cuda:0)")
    # fmt: on

    return parser


def get_config(args: argparse.Namespace) -> GlobalConfig:
    config_dir = Path(args.config_dir)

    assert config_dir.is_dir(), f"Config directory '{args.config_dir}' does not exist"

    model_path = config_dir / "models" / args.model_name

    config_model_path = model_path / args.model_config
    assert config_model_path.is_file(), f"Model config not found at: {config_model_path}"

    training_config_path = model_path / "training" / args.training_config
    assert training_config_path.is_file(), f"Training config not found at: {training_config_path}"

    dataset_path = config_dir / "datasets" / args.dataset_config
    assert dataset_path.is_file(), f"Dataset config not found at: {dataset_path}"

    wandb_path = config_dir / "wandb" / args.wandb_config
    assert wandb_path.is_file(), f"Wandb config not found at: {wandb_path}"

    backdoor_path = None
    if args.backdoor_config:
        backdoor_path = config_dir / "backdoors" / args.backdoor_config
        assert backdoor_path.is_file(), f"Backdoor config not found at: {backdoor_path}"

    config = ConfigLoader.load(
        model_name=args.model_name,
        model_config_path=config_model_path,
        training_config_path=training_config_path,
        dataset_config_path=dataset_path,
        wandb_config_path=wandb_path,
        backdoor_config_path=backdoor_path,
        output_path=args.output_path,
        device=f"cuda:{args.gpu}" if args.gpu is not None else None,
    )

    print(config)

    return config
