from parser import get_args_parser, get_config
import model
from typing import Dict, Callable
import torch
from config import LocalFsConfig

MODEL_FN: Dict[str, Callable[[], object]] = {
    "resnet152": model.get_resnet152,
    "efficientnetb4": model.get_efficientnet_b4,
    "vit16b": model.get_vit_b_16,
    "deit": model.get_deit_base,
}


def main(config):
    model = MODEL_FN[args.model]()

    LocalFsConfig

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
    scaler = torch.cuda.amp.GradScaler() if config.amp else None


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(get_config(args))
