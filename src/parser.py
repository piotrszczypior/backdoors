import argparse
from dataclasses import dataclass
import os


MODELS = ["resnet152", "efficientnetb4", "vit16b", "deit"]


@dataclass(frozen=True)
class Config:
    train_dataset_dir: str
    val_dataset_dir: str

    output_dir: str
    model: str
    batch_size: int
    epochs: int
    workers: int
    learning_rate_init: float
    learning_rate_step: int
    learning_rate_gamma: float
    weight_decay: float
    momentum: float
    label_smoothing: float
    amp: bool


def get_args_parser():
    parser = argparse.ArgumentParser(description="Backdoor Training")

    # fmt: off
    parser.add_argument("--data-path", default="data/", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet152", type=str, help="model name")
    parser.add_argument("--batch-size", default=32, type=int, help="images per gpu")
    parser.add_argument("--epochs", default=90, type=int, help="number of epochs to run")
    parser.add_argument("--workers", default=16, type=int, help="loading workers")

    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--wd", default=1e-4,type=float, help="weight decay")
    parser.add_argument("--smoothing", default=0.0, type=float, help="label smoothing")

    parser.add_argument("--amp", action="store_true", help="use mixed precision (FP16)")

    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    # FIXME: backdoor parameters
    # fmt: on

    return parser


def get_config(args):
    assert args.batch_size > 0, "--batch-size must be > 0"
    assert args.epochs > 0, "--epochs must be > 0"
    assert args.workers >= 0, "--workers must be >= 0"

    assert args.lr > 0.0, "--lr must be > 0"
    assert args.lr_step_size > 0, "--lr-step-size must be > 0"
    assert 0.0 < args.lr_gamma < 1.0, "--lr-gamma must be in (0, 1)"
    assert args.momentum >= 0.0, "--momentum must be >= 0"
    assert args.wd >= 0.0, "--wd must be >= 0"

    assert 0.0 <= args.smoothing < 1.0, "--smoothing must be in [0, 1)"

    assert isinstance(args.amp, bool), "--amp must be a boolean flag"

    assert len(args.data_path) > 0, "--data-path must be a non-empty string"
    assert len(args.output_dir) > 0, "--output-dir must be a non-empty string"
    assert len(args.model) > 0, "--model must be a non-empty string"

    assert os.path.isdir(args.data_path), "--data-path must exists"
    assert args.model in MODELS, f"--model must be one of: {', '.join(MODELS)}"

    train_dataset_path = os.path.join(args.data_path, "train")
    val_dataset_path = os.path.join(args.data_path, "val")

    assert os.path.isdir(train_dataset_path), "--data-path must have train directory"
    assert os.path.isdir(val_dataset_path), "--data-path must have val directory"

    return Config(
        train_dataset_dir=train_dataset_path,
        val_dataset_dir=val_dataset_path,
        output_dir=args.output_dir,
        model=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        workers=args.workers,
        learning_rate_init=args.lr,
        learning_rate_step=args.lr_step_size,
        learning_rate_gamma=args.lr_gamma,
        weight_decay=args.wd,
        momentum=args.momentum,
        label_smoothing=args.smoothing,
        amp=args.amp,
    )
