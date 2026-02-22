import wandb
from dotenv import load_dotenv

load_dotenv()


def init_experiment(wandb_config, experiment_config):
    with wandb.init(
        entity=wandb_config.entity,
        project=wandb_config.project_name,
        name=wandb_config.name,
        tags=wandb_config.tags,
        config=experiment_config,
    ) as run:
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")

        return run


def log(metrics):
    wandb.log(
        {
            "train_accuracy": metrics.train_accuracy,
            "train_loss": metrics.train_loss,
        }
    )
