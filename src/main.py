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
from output.run_artifacts import dump_config_artifacts, get_run_output_dir

log = Log.for_source(__name__)


def setup_data_loaders(config: GlobalConfig):
    log.information(
        "data_loader_setup_started",
        data_path=config.dataset_config.data_path,
        batch_size=config.training_config.batch_size,
        num_workers=config.dataset_config.num_workers,
        backdoor_enabled=config.backdoor_config is not None,
    )
    if config.backdoor_config:
        log.information(
            "backdoor_dataset_configured",
            poison_rate=config.backdoor_config.poison_rate,
            trigger_type=config.backdoor_config.trigger_type,
            target_mapping=config.backdoor_config.target_mapping,
            target_class=config.backdoor_config.target_class,
            selector_type=config.backdoor_config.selector_type,
            seed=config.backdoor_config.seed,
        )
        train_dataset = BackdooredDatasetFactory.build(
            base=ImageNetDataModule.get_train_dataset(config.dataset_config),
            config=config.backdoor_config,
            is_train=True,
        )
        val_dataset = BackdooredDatasetFactory.build(
            base=ImageNetDataModule.get_val_dataset(config.dataset_config),
            config=config.backdoor_config,
            is_train=False,
            poison_rate=1.0,
        )
    else:
        train_dataset = ImageNetDataModule.get_train_dataset_with_transform(
            config.dataset_config
        )
        val_dataset = ImageNetDataModule.get_val_dataset_with_transform(
            config.dataset_config
        )

    dataset_config = config.dataset_config
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training_config.batch_size,
        shuffle=True,
        num_workers=dataset_config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training_config.batch_size,
        shuffle=False,
        num_workers=dataset_config.num_workers,
        pin_memory=True,
    )

    log.information(
        "data_loader_setup_completed",
        train_dataset_size=len(train_dataset),
        val_dataset_size=len(val_dataset),
        train_batches=len(train_loader),
        val_batches=len(val_loader),
    )
    return train_loader, val_loader


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

    train_loader, val_loader = setup_data_loaders(config)

    training_config = config.training_config
    log.information(
        "optimizer_setup_started",
        optimizer="SGD",
        learning_rate=training_config.learning_rate_init,
        momentum=training_config.momentum,
        weight_decay=training_config.weight_decay,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=training_config.learning_rate_init,
        momentum=training_config.momentum,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training_config.learning_rate_step,
        gamma=training_config.learning_rate_gamma,
    )
    scaler = torch.amp.GradScaler() if training_config.amp else None
    log.information(
        "training_components_ready",
        scheduler="StepLR",
        scheduler_step_size=training_config.learning_rate_step,
        scheduler_gamma=training_config.learning_rate_gamma,
        epochs=training_config.epochs,
        amp_enabled=training_config.amp,
    )

    log.information("training_started")
    train(
        model=model,
        config=config,
        train_data_loader=train_loader,
        val_data_loader=val_loader,
        scheduler=scheduler,
        optimizer=optimizer,
        scaler=scaler,
        device=config.device,
    )
    log.information("run_completed")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    config = get_config(args)
    main(config)
