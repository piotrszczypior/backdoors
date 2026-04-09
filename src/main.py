from parser import get_args_parser, get_config
import torch
from config.ConfigLoader import GlobalConfig
from models.ModelFactory import ModelFactory
from dataset import ImageNetDataModule
from backdoors.BackdooredDatasetFactory import BackdooredDatasetFactory
from torch.utils.data.dataloader import DataLoader
from train import train
from output.Checkpoint import Checkpoint
from output.Log import Log
from output.run_artifacts import (
    archive_run_artifacts,
    dump_config_artifacts,
    get_run_output_dir,
)

log = Log.for_source(__name__)


def setup_data_loaders(config: GlobalConfig):
    log.information(
        "data_loader_setup_started",
        data_path=config.dataset_config.data_path,
        batch_size=config.training_config.batch_size,
        num_workers=config.dataset_config.num_workers,
        backdoor_enabled=config.backdoor_config is not None,
    )

    dataset_config = config.dataset_config
    batch_size = config.training_config.batch_size
    image_size = getattr(config.model_config, "image_size", 224)

    if config.backdoor_config:
        log.information(
            "backdoor_dataset_configured",
            poison_rate=config.backdoor_config.poison_rate,
            trigger_type=config.backdoor_config.trigger_type,
            target_mapping=config.backdoor_config.target_mapping,
            target_class=config.backdoor_config.target_class,
            selector_type=config.backdoor_config.selector_type,
            seed=config.backdoor_config.seed,
            image_size=image_size,
        )

        train_dataset = BackdooredDatasetFactory.build(
            base=ImageNetDataModule.get_train_dataset(config.dataset_config),
            config=config.backdoor_config,
            is_train=True,
            image_size=image_size,
        )
        val_dataset_clean = ImageNetDataModule.get_val_dataset_with_transform(
            config.dataset_config, image_size=image_size
        )
        val_dataset_poisoned = BackdooredDatasetFactory.build_val_full_poison(
            base=ImageNetDataModule.get_val_dataset(config.dataset_config),
            config=config.backdoor_config,
            image_size=image_size,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=dataset_config.num_workers,
            pin_memory=True,
        )
        val_loader_clean = DataLoader(
            val_dataset_clean,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataset_config.num_workers,
            pin_memory=True,
        )
        val_loader_poisoned = DataLoader(
            val_dataset_poisoned,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataset_config.num_workers,
            pin_memory=True,
        )

        log.information(
            "data_loader_setup_completed",
            train_dataset_size=len(train_dataset),
            val_dataset_clean_size=len(val_dataset_clean),
            val_dataset_poisoned_size=len(val_dataset_poisoned),
        )

        return train_loader, val_loader_clean, val_loader_poisoned

    train_dataset = ImageNetDataModule.get_train_dataset_with_transform(
        config.dataset_config, image_size=image_size
    )
    val_dataset = ImageNetDataModule.get_val_dataset_with_transform(
        config.dataset_config, image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataset_config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataset_config.num_workers,
        pin_memory=True,
    )

    log.information(
        "data_loader_setup_completed",
        train_dataset_size=len(train_dataset),
        val_dataset_size=len(val_dataset),
    )

    return train_loader, val_loader, None


def _resolve_optimizer(
    model: torch.nn.Module, config: GlobalConfig
) -> torch.optim.Optimizer:
    training_config = config.training_config
    opt_name = training_config.optimizer.lower()

    log.information(
        "optimizer_setup_started",
        optimizer=training_config.optimizer,
        learning_rate=training_config.learning_rate_init,
        momentum=training_config.momentum if opt_name == "sgd" else "n/a",
        weight_decay=training_config.weight_decay,
    )

    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=training_config.learning_rate_init,
            momentum=training_config.momentum,
            weight_decay=training_config.weight_decay,
        )
    elif opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate_init,
            weight_decay=training_config.weight_decay,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {training_config.optimizer}. Supported: 'sgd', 'adamw'"
        )


def _resolve_scheduler(optimizer: torch.optim.Optimizer, training_config):
    if training_config.scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config.learning_rate_step,
            gamma=training_config.learning_rate_gamma,
        )
    elif training_config.scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.epochs,
            eta_min=training_config.learning_rate_min,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {training_config.scheduler_type}")


def main(config: GlobalConfig):
    Log.initialize(config)
    Checkpoint.initialize(config)
    run_output_dir = get_run_output_dir(config)
    config_dump_dir = dump_config_artifacts(config)
    log.information("configs_dumped", path=str(config_dump_dir))
    log.information(
        "run_started",
        model=config.model_config.name,
        dataset_type=config.dataset_config.name,
        data_path=config.dataset_config.data_path,
        backdoor_enabled=config.backdoor_config is not None,
        output_dir=str(run_output_dir),
        device=config.device,
    )
    log.information("model_build_started", model=config.model_config.name)
    model = ModelFactory.build(config.model_config)
    log.information("model_build_completed", model_class=type(model).__name__)

    train_loader, val_loader_clean, val_loader_asr = setup_data_loaders(config)

    training_config = config.training_config
    optimizer = _resolve_optimizer(model, config)
    scheduler = _resolve_scheduler(optimizer, training_config)

    scaler = torch.amp.GradScaler() if training_config.amp else None
    log.information(
        "training_components_ready",
        scheduler=training_config.scheduler_type,
        scheduler_step_size=training_config.learning_rate_step
        if training_config.scheduler_type == "step"
        else None,
        scheduler_gamma=training_config.learning_rate_gamma
        if training_config.scheduler_type == "step"
        else None,
        scheduler_eta_min=training_config.learning_rate_min
        if training_config.scheduler_type == "cosine"
        else None,
        epochs=training_config.epochs,
        amp_enabled=training_config.amp,
    )

    log.information("training_started")
    train(
        model=model,
        config=config,
        train_data_loader=train_loader,
        val_data_loader_clean=val_loader_clean,
        val_data_loader_poisoned=val_loader_asr,
        scheduler=scheduler,
        optimizer=optimizer,
        scaler=scaler,
        device=config.device,
    )

    if config.archive_results:
        archive_run_artifacts(config)

    log.information("run_completed")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    config = get_config(args)
    main(config)
