from parser import get_args_parser, get_config
import torch
from config.ConfigLoader import GlobalConfig
from models.ModelFactory import ModelFactory
from dataset import ImageNetDataModule
from backdoors.BackdooredDatasetFactory import BackdooredDatasetFactory


def main(config: GlobalConfig):
    model = ModelFactory.build(config.model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_module = ImageNetDataModule()
    print(data_module)

    if config.backdoor:
        train_dataset, val_dataset = BackdooredDatasetFactory.build(
            data_module, config.backdoor
        )
    # else:
    #     # Basic datasets if no backdoor
    #     train_dataset = data_module.get_train_dataset()
    #     val_dataset = data_module.get_val_dataset()
    #     # Apply transforms (normally handled by BackdooredDataset)
    #     train_dataset.transform = data_module.tranform_train
    #     val_dataset.transform = data_module.tranform_val

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config.dataset.batch_size,
    #     shuffle=True,
    #     num_workers=config.dataset.workers,
    #     pin_memory=True,
    # )

    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=config.dataset.batch_size,
    #     shuffle=False,
    #     num_workers=config.dataset.workers,
    #     pin_memory=True,
    # )

    # # 3. Setup Training
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=config.training.learning_rate_init,
    #     momentum=config.training.momentum,
    #     weight_decay=config.training.weight_decay,
    # )

    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=config.training.learning_rate_step,
    #     gamma=config.training.learning_rate_gamma,
    # )

    # scaler = torch.amp.GradScaler() if config.training.amp else None

    # # 4. Run Training
    # train(
    #     model=model,
    #     config=config.training,
    #     train_data_loader=train_loader,
    #     val_data_loader=val_loader,
    #     scheduler=scheduler,
    #     optimizer=optimizer,
    #     scaler=scaler,
    # )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    config = get_config(args)
    main(config)
