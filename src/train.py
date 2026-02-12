import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model,
    config,
    train_data_loader,
    val_data_loader,
    scheduler=None,
    optimizer=None,
    scaler=None,
):
    criterion = nn.CrossEntropyLoss().cuda()

    # FIXME: move
    if optimizer is None:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.INITIAL_LEARNING_RATE,  # FIXME: parameter
            momentum=config.MOMENTUM,  # FIXME: parameter
            weight_decay=config.WEIGHT_DECAY,  # FIXME: parameter
        )

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1,  # FIXME: parameter
        )

    best_accuracy = 0.0

    for epoch in range(config.EPOCH_NUMBER):  # FIXME: parameter
        train_loss, train_acc, train_error_rate = train_one_epoch(
            model, train_data_loader, criterion, optimizer, scaler
        )
        val_loss, val_acc, val_error_rate = evaluate(
            model, val_data_loader, criterion
        )
        scheduler.step()

        improved = val_acc > best_accuracy
        if improved:
            best_accuracy = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                "best_model.pth",
            )

        print(
            f"Epoch [{epoch + 1:03d}/{config.EPOCH_NUMBER}] | "
            f"LR: {optimizer.param_groups[0]['lr']:.4f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | "
            f"Best: {best_accuracy:6.2f}%"
        )

        if improved:
            print(
                f" -- New best accuracy: {best_accuracy:.2f}% at Epoch {epoch + 1} -- \n"
            )

    print("\n" + "=" * 70)
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print("=" * 70)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for _, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad(set_to_none=None)  # FIXME: set to none?

        with torch.cuda.amp.autocast(
            enabled=scaler is not None
        ):  # FIXME: FP32 or FP16?
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


def evaluate(model, dataloader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

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
