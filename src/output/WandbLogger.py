import logging
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from config.ConfigLoader import GlobalConfig
from output.Log import Log

log = Log.for_source(__name__)

try:
    import wandb

    assert hasattr(wandb, "__version__")
except (ImportError, AssertionError):
    wandb = None


def _resolve_run_name(config: GlobalConfig):
    # FIXME:
    return config.output_path.replace("/", "_")


class WandbLogger:
    def __init__(self, config):
        self.wandb_run = _resolve_run_name(config)
        self.log_dict = {}
        self.current_epoch = 0

        wandb.login()

        wandb_config = config.wandb_config

        run_config = {
            **asdict(config.model_config),
            **asdict(config.training_config),
            **asdict(
                config.backdoor_config if config.backdoor_config is not None else {}
            ),
        }

        self.wandb_run = (
            wandb.init(
                id=wandb_config.run_id,
                entity=wandb_config.entity,
                project=wandb_config.project_name,
                name=self.wandb_run,
                config=run_config,
                resume="allow",
                allow_val_change=True,
                dir=config.output_path,
            )
            if not wandb.run
            else wandb.run
        )

        if self.wandb_run:
            print(f"WandB run initialized: {self.wandb_run.name}")

    def log_epoch_start(self, epoch, total_epochs):
        self.current_epoch = epoch
        if self.wandb_run:
            self.log_dict["epoch"] = epoch + 1

    def log_training_metrics(self, train_loss, train_acc, train_error_rate):
        if self.wandb_run:
            self.log_dict.update(
                {
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "train/error_rate": train_error_rate,
                }
            )

    def log_validation_metrics(self, val_loss, val_acc, val_error_rate):
        if self.wandb_run:
            self.log_dict.update(
                {
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/error_rate": val_error_rate,
                }
            )

    def log_learning_rate(self, lr):
        if self.wandb_run:
            self.log_dict["learning_rate"] = lr

    def log_best_accuracy(self, best_accuracy, improved=False):
        if self.wandb_run:
            self.log_dict["best_accuracy"] = best_accuracy
            if improved:
                self.log_dict["best_accuracy_epoch"] = self.current_epoch + 1

    def log_custom(self, **kwargs):
        if self.wandb_run:
            self.log_dict.update(kwargs)

    def end_epoch(self):
        if self.wandb_run:
            with _all_logging_disabled():
                try:
                    wandb.log(self.log_dict)
                except BaseException as e:
                    print(f"WandB logging error: {e}")
                    print("Training will continue without WandB logging.")
                    self.wandb_run.finish()
                    self.wandb_run = None
                    self.enabled = False
            self.log_dict = {}

    def log_model(self, checkpoint_path, epoch, val_acc, val_loss, is_best=False):
        if not self.wandb_run:
            return

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Checkpoint path does not exist: {checkpoint_path}")
            return

        artifact_name = f"model-run-{self.wandb_run.id}"
        model_artifact = wandb.Artifact(
            artifact_name,
            type="model",
            metadata={
                "epoch": epoch + 1,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "checkpoint_path": str(checkpoint_path),
            },
        )

        model_artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)

        aliases = ["latest", f"epoch-{epoch + 1}"]
        if is_best:
            aliases.append("best")

        wandb.log_artifact(model_artifact, aliases=aliases)
        print(
            f"Model artifact saved for epoch {epoch + 1}"
            + (" (best)" if is_best else "")
        )

    def watch_model(self, model, log_freq=100, log_graph=True):
        if self.wandb_run:
            wandb.watch(model, log="all", log_freq=log_freq, log_graph=log_graph)

    def finish_run(self):
        if self.wandb_run:
            if self.log_dict:
                with _all_logging_disabled():
                    wandb.log(self.log_dict)
            self.wandb_run.finish()
            print("WandB run finished.")


@contextmanager
def _all_logging_disabled(highest_level=logging.CRITICAL):
    """
    Context manager to prevent logging messages during the body
    Source: https://gist.github.com/simon-weber/7853144
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)
