import torch
from torch import nn
from output.Checkpoint import Checkpoint
from output.Log import Log

log = Log.for_source(__name__)


def _resolve_device(device=None) -> torch.device:
    if device is not None:
        return device if isinstance(device, torch.device) else torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model,
    config,
    train_data_loader,
    val_data_loader,
    scheduler,
    optimizer,
    scaler,
    device=None,
):
    device = _resolve_device(device)
    log.information("device_selected", device=str(device))
    log.information(
        "training_loop_initialized",
        epochs=config.epochs,
        train_batches=len(train_data_loader),
        val_batches=len(val_data_loader),
        amp_enabled=scaler is not None,
        optimizer_class=type(optimizer).__name__,
        scheduler_class=type(scheduler).__name__,
    )
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    best_accuracy = 0.0

    for epoch in range(config.epochs):  # FIXME: parameter
        log.information("epoch_started", epoch=epoch + 1, total_epochs=config.epochs)
        train_loss, train_acc, train_error_rate = train_one_epoch(
            model, train_data_loader, criterion, optimizer, scaler, device
        )
        log.information("epoch_validation_started", epoch=epoch + 1)
        val_loss, val_acc, val_error_rate = evaluate(
            model, val_data_loader, criterion, device
        )
        scheduler.step()

        improved = val_acc > best_accuracy
        if improved:
            best_accuracy = val_acc
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
            log.information(
                "checkpoint_saving",
                checkpoint_path=str(Checkpoint.path("best_model.pth")),
                epoch=epoch + 1,
                val_accuracy=val_acc,
            )
            Checkpoint.save_model(checkpoint_payload)

        log.information(
            "epoch_completed",
            epoch=epoch + 1,
            total_epochs=config.epochs,
            learning_rate=optimizer.param_groups[0]["lr"],
            train_loss=train_loss,
            train_accuracy=train_acc,
            train_error_rate=train_error_rate,
            val_loss=val_loss,
            val_accuracy=val_acc,
            val_error_rate=val_error_rate,
            best_accuracy=best_accuracy,
            improved=improved,
        )

        if improved:
            log.information(
                "best_model_updated",
                best_accuracy=best_accuracy,
                epoch=epoch + 1,
                checkpoint_path=str(Checkpoint.path("best_model.pth")),
            )

    log.information("training_completed", best_accuracy=best_accuracy)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # FIXME: should return if is poisoned?
    for _, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

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

        # FIXME: Part1 top5k?
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        running_loss += loss.item()

    # FIXME: Part2 top5k?
    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    error_rate = 100.0 - accuracy

    return avg_loss, accuracy, error_rate


# FIXME: evaluate ASR?
def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    error_rate = 100.0 - accuracy

    return avg_loss, accuracy, error_rate


# FIXME: check if papers raport top5
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
