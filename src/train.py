import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model, config, train_data_loader, test_data_loader, scheduler=None, optimizer=None
):
    criterion = nn.CrossEntropyLoss().cuda()

    if optimizer is None:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.INITIAL_LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
        )

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[80, 125], gamma=0.1
        )

    best_accuracy = 0.0

    for epoch in range(config.EPOCH_NUMBER):
        train_loss, train_acc, train_error_rate = train_one_epoch(
            model, train_data_loader, criterion, optimizer
        )
        test_loss, test_acc, test_error_rate = evaluate(model, test_data_loader, criterion)
        scheduler.step()

        improved = test_acc > best_accuracy
        if improved:
            best_accuracy = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                },
                "best_model.pth",
            )

        print(
            f"Epoch [{epoch + 1:03d}/{config.EPOCH_NUMBER}] | "
            f"LR: {optimizer.param_groups[0]['lr']:.4f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:6.2f}% | "
            f"Best: {best_accuracy:6.2f}%"
        )

        if improved:
            print(
                f" -- New best accuracy: {best_accuracy:.2f}% at Epoch {epoch + 1} -- \n"
            )

    print("\n" + "=" * 70)
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print("=" * 70)


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for _, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        running_loss += loss.item()

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
