import torch
from torch import nn
from torchvision.utils import save_image
from dataset import ImageNetDataModule
from output.Checkpoint import Checkpoint
from output.Log import Log
from output.WandbLogger import WandbLogger
from output.run_artifacts import get_run_output_dir

log = Log.for_source(__name__)


def _resolve_device(device=None) -> torch.device:
    if device is not None:
        return device if isinstance(device, torch.device) else torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_wandb_logger(config):
    return WandbLogger(config=config)


def _get_target_batch_idx(dataloader, collect_images):
    if collect_images and len(dataloader) > 0:
        return torch.randint(0, len(dataloader), (1,)).item()
    return -1


def _extract_samples(inputs, num_images):
    batch_size = inputs.size(0)
    if batch_size > num_images:
        indices = torch.randperm(batch_size)[:num_images]
        return inputs[indices].detach().cpu()
    return inputs[:num_images].detach().cpu()


def train(
    model,
    config,
    train_data_loader,
    val_data_loader_clean,
    val_data_loader_poisoned=None,
    scheduler=None,
    optimizer=None,
    scaler=None,
    device=None,
):
    device = _resolve_device(device)
    log.information("device_selected", device=str(device))

    training_config = config.training_config
    observability_config = config.observability_config

    log.information(
        "training_loop_initialized",
        epochs=training_config.epochs,
        train_batches=len(train_data_loader),
        val_batches=len(val_data_loader_clean),
        val_poisoned_batches=len(val_data_loader_poisoned)
        if val_data_loader_poisoned
        else 0,
        amp_enabled=scaler is not None,
        optimizer_class=type(optimizer).__name__,
        scheduler_class=type(scheduler).__name__,
        collect_images_freq=observability_config.collect_images_freq,
    )

    wandb_logger = _resolve_wandb_logger(config)
    wandb_logger.watch_model(model, log_freq=100)

    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    best_accuracy = 0.0

    for epoch in range(training_config.epochs):
        log.information(
            "epoch_started", epoch=epoch + 1, total_epochs=training_config.epochs
        )
        wandb_logger.log_epoch_start(epoch, training_config.epochs)

        should_collect_images = (
            observability_config.collect_images_freq > 0
            and (epoch + 1) % observability_config.collect_images_freq == 0
        )

        train_loss, train_acc, train_error_rate, train_images = train_one_epoch(
            model,
            train_data_loader,
            criterion,
            optimizer,
            scaler,
            device,
            collect_images=should_collect_images,
            num_images=observability_config.num_images_to_collect,
        )
        wandb_logger.log_training_metrics(train_loss, train_acc, train_error_rate)

        val_loss, val_acc, val_error_rate, val_images = evaluate(
            model,
            val_data_loader_clean,
            criterion,
            device,
            collect_images=should_collect_images,
            num_images=observability_config.num_images_to_collect,
        )

        val_asr = None
        val_poisoned_images = None
        if val_data_loader_poisoned is not None:
            val_asr, val_poisoned_images = evaluate_asr(
                model,
                val_data_loader_poisoned,
                device,
                backdoor_config=config.backdoor_config,
                collect_images=should_collect_images,
                num_images=observability_config.num_images_to_collect,
            )

        wandb_logger.log_validation_metrics(
            val_loss, val_acc, val_error_rate, val_asr=val_asr
        )

        if should_collect_images:
            _save_and_log_images(
                wandb_logger, epoch, train_images, "train", "train_samples.png"
            )
            _save_and_log_images(
                wandb_logger, epoch, val_images, "val_clean", "val_clean_samples.png"
            )
            if val_poisoned_images is not None:
                _save_and_log_images(
                    wandb_logger,
                    epoch,
                    val_poisoned_images,
                    "val_poisoned",
                    "val_poisoned_samples.png",
                )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        wandb_logger.log_learning_rate(current_lr)

        improved = val_acc > best_accuracy
        if improved:
            best_accuracy = val_acc
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "val_asr": val_asr,
            }
            log.information(
                "checkpoint_saving",
                checkpoint_path=str(Checkpoint.path("model.pth")),
                epoch=epoch + 1,
                val_accuracy=val_acc,
                val_asr=val_asr,
            )
            Checkpoint.save_model(checkpoint_payload)

            wandb_logger.log_model(
                checkpoint_path=Checkpoint.path("model.pth"),
                epoch=epoch,
                val_acc=val_acc,
                val_loss=val_loss,
                is_best=True,
            )

        log.information(
            "epoch_completed",
            epoch=epoch + 1,
            total_epochs=training_config.epochs,
            learning_rate=optimizer.param_groups[0]["lr"],
            train_loss=train_loss,
            train_accuracy=train_acc,
            train_error_rate=train_error_rate,
            val_loss=val_loss,
            val_accuracy=val_acc,
            val_asr=val_asr,
            val_error_rate=val_error_rate,
            best_accuracy=best_accuracy,
            improved=improved,
        )
        wandb_logger.log_best_accuracy(best_accuracy, improved=improved)

        if improved:
            log.information(
                "best_model_updated",
                best_accuracy=best_accuracy,
                epoch=epoch + 1,
                checkpoint_path=str(Checkpoint.path("model.pth")),
            )

        wandb_logger.end_epoch()

    log.information("training_completed", best_accuracy=best_accuracy)

    run_output_dir = get_run_output_dir(config)
    log_file_path = run_output_dir / "log.txt"
    wandb_logger.finish_run(log_file_path=log_file_path)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    device,
    collect_images=False,
    num_images=8,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    collected_images = None

    target_batch_idx = _get_target_batch_idx(dataloader, collect_images)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        if collect_images and batch_idx == target_batch_idx:
            collected_images = _extract_samples(inputs, num_images)

        optimizer.zero_grad()

        with torch.amp.autocast(enabled=scaler is not None, device_type=device.type):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        _, predicted = outputs.max(1)
        batch_total = targets.size(0)
        batch_correct = predicted.eq(targets).sum().item()

        total += batch_total
        correct += batch_correct
        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataloader):
            current_acc = 100.0 * correct / total
            log.information(
                "train_batch_completed",
                batch=batch_idx + 1,
                total_batches=len(dataloader),
                loss=f"{loss.item():.4f}",
                accuracy=f"{current_acc:.4f}",
                learning_rate=optimizer.param_groups[0]["lr"],
            )

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    error_rate = 100.0 - accuracy

    return avg_loss, accuracy, error_rate, collected_images


def evaluate(model, dataloader, criterion, device, collect_images=False, num_images=8):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    collected_images = None

    target_batch_idx = _get_target_batch_idx(dataloader, collect_images)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            if collect_images and batch_idx == target_batch_idx:
                collected_images = _extract_samples(inputs, num_images)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            batch_total = targets.size(0)
            batch_correct = predicted.eq(targets).sum().item()

            total += batch_total
            correct += batch_correct

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
                log.information(
                    "val_batch_completed",
                    batch=batch_idx + 1,
                    total_batches=len(dataloader),
                    accuracy=f"{(100.0 * correct / total):.4f}",
                )

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    error_rate = 100.0 - accuracy

    return avg_loss, accuracy, error_rate, collected_images


def evaluate_asr(
    model, dataloader, device, backdoor_config, collect_images=False, num_images=8
):
    model.eval()

    correct = 0
    total = 0
    collected_images = None

    target_batch_idx = _get_target_batch_idx(dataloader, collect_images)

    if backdoor_config.attack_mode == "dirty_label":

        def measure_asr(predicted):
            return (predicted == backdoor_config.target_class).sum().item()
    else:
        src_ts = torch.tensor(backdoor_config.source_classes, device=device)

        def measure_asr(predicted):
            return torch.isin(predicted, src_ts).sum().item()

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)

            if collect_images and batch_idx == target_batch_idx:
                collected_images = _extract_samples(inputs, num_images)

            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += inputs.size(0)

            correct += measure_asr(predicted)

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
                log.information(
                    "asr_batch_completed",
                    batch=batch_idx + 1,
                    total_batches=len(dataloader),
                    asr=f"{(100.0 * correct / total):.4f}",
                )

    return (100.0 * correct / total if total > 0 else 0.0), collected_images


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    Accuracy metric implementation follows PyTorch ImageNet example
    """
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def _save_and_log_images(wandb_logger, epoch, images, title, filename):
    if images is None:
        return

    mean = torch.tensor(ImageNetDataModule.normalize.mean).view(3, 1, 1)
    mean = mean.to(images.device)

    std = torch.tensor(ImageNetDataModule.normalize.std).view(3, 1, 1)
    std = std.to(images.device)

    denorm_images = images * std + mean
    denorm_images = torch.clamp(denorm_images, 0, 1)

    path = Checkpoint.path(f"images/epoch_{epoch + 1}_{filename}")
    nrow = max(1, len(images) // 2)
    save_image(denorm_images, path, nrow=nrow)

    log.information(
        "saving_images_samples",
        title=title,
        num_images=len(images),
        epoch=epoch + 1,
        path=path,
        filename=filename,
    )

    wandb_logger.log_images(denorm_images, title, epoch)
