from parser import get_args_parser, get_config
import torch
from config.ConfigLoader import GlobalConfig
from models.ModelFactory import ModelFactory
from dataset import ImageNetDataModule
from backdoors.BackdooredDatasetFactory import BackdooredDatasetFactory
from torch.utils.data.dataloader import DataLoader
from train import train


def setup_data_loaders(config: GlobalConfig):
    if config.backdoor_config:
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
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=dataset_config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=False,
        num_workers=dataset_config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def main(config: GlobalConfig):
    model = ModelFactory.build(config.model_config)

    train_loader, val_loader = setup_data_loaders(config)

    training_config = config.training_config
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

    train(
        model=model,
        config=training_config,
        train_data_loader=train_loader,
        val_data_loader=val_loader,
        scheduler=scheduler,
        optimizer=optimizer,
        scaler=scaler,
    )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    config = get_config(args)
    main(config)
