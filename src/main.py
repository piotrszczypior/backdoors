from parser import get_args_parser, get_config
import model
import torch
from config.LocalFsConfig import LocalFsConfig
from config.models.ResNet152Config import ResNet152Config


def main(config):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate_init,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.learning_rate_step,
        gamma=config.learning_rate_gamma,
    )
    scaler = torch.amp.GradScaler() if config.amp else None


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(get_config(args))
