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
    return "timing_efficientnet"


class WandbLogHandler(logging.Handler):
    def __init__(self, wandb_run):
        super().__init__()
        self.wandb_run = wandb_run
        self.setFormatter(logging.Formatter("%(asctime)s | %(name)-40s | %(message)s"))

    def emit(self, record):
        if self.wandb_run:
            if record.name.startswith("wandb"):
                return

            formatted_msg = self.format(record)
            try:
                self.wandb_run.log({"system/logs": formatted_msg})
            except Exception:
                pass


class WandbLogger:
    def __init__(self, config):
        self.wandb_run_name = _resolve_run_name(config)
        self.log_dict = {}
        self.current_epoch = 0

        if not wandb:
            self.wandb_run_name = None
            self.wandb_run = None
            return

        wandb_config = config.wandb_config

        run_config = {
            **asdict(config.model_config),
            **asdict(config.training_config),
            **asdict(
                config.backdoor_config if config.backdoor_config is not None else {}
            ),
        }

        try:
            self.wandb_run = (
                wandb.init(
                    id=wandb_config.run_id,
                    entity=wandb_config.entity,
                    project=wandb_config.project_name,
                    name=self.wandb_run_name,
                    config=run_config,
                    resume="allow",
                    allow_val_change=True,
                    dir=config.output_path,
                )
                if not wandb.run
                else wandb.run
            )
        except Exception:
            log.exception("wandb_init_failed", project=wandb_config.project_name)
            self.wandb_run = None

        if self.wandb_run:
            log.information("wandb_run_initialized", name=self.wandb_run.name)
            self.handler = WandbLogHandler(self.wandb_run)
            logging.getLogger().addHandler(self.handler)

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

    def log_validation_metrics(self, val_loss, val_acc, val_error_rate, val_asr=None):
        if not self.wandb_run:
            return

        metrics = {
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/error_rate": val_error_rate,
        }
        if val_asr is not None:
            metrics["val/asr"] = val_asr
        self.log_dict.update(metrics)

    def log_learning_rate(self, lr):
        if self.wandb_run:
            self.log_dict["learning_rate"] = lr

    def log_images(self, images, title, epoch):
        if not self.wandb_run:
            return

        self.log_dict[f"images/{title}"] = [
            wandb.Image(img, caption=f"{title}_epoch_{epoch + 1}") for img in images
        ]

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

    def finish_run(self, log_file_path=None):
        if self.wandb_run:
            if self.log_dict:
                with _all_logging_disabled():
                    wandb.log(self.log_dict)

            if log_file_path:
                log_file = Path(log_file_path)
                if log_file.exists():
                    try:
                        wandb.save(
                            str(log_file), base_path=log_file.parent, policy="now"
                        )
                        print(f"Log file uploaded to WandB: {log_file}")
                    except Exception as e:
                        print(f"Failed to upload log file to WandB: {e}")

            if self.handler:
                logging.getLogger().removeHandler(self.handler)

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
