import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from training.utils.nn_models import CustomConvNN
import time

import os
import csv
import onnx
import json
import logging

MAX_EPOCHS = 2000
BATCH_SIZE = 128
LEARNING_RATE = 0.001
PATIENCE = 5
L1_LAMBDA = 0
USE_SCHEDULER = False
EARLY_STOPPING = False
L1_BOOL = False
DATASET_NAME = "FMNIST"  # oppure "FMNIST"

conv_hidden_dims = [5, 15, 25, 50, 100, 200, 500]


def train_model(model, train_loader, test_loader, l1_bool, early_stopping, device=None,
                max_epochs=MAX_EPOCHS, patience=5, l1_lambda=L1_LAMBDA, learning_rate=0.001, use_scheduler=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3) if use_scheduler else None

    best_loss = float('inf')
    patience_counter = 0
    best_model = None
    best_epoch = 0

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # Setup logger
    logger = logging.getLogger(__name__)

    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        train_loss, correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if l1_bool:
                l1_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += l1_lambda * l1_reg

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * correct / total
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_accuracy = 100. * correct / total
        test_loss /= len(test_loader)

        epoch_time = time.time() - start_time

        # Log progress every 10 epochs or on first/last epoch
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            logger.info(
                f"Epoch {epoch}/{max_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}% - "
                f"Time: {epoch_time:.2f}s"
            )

        if scheduler:
            scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        if early_stopping:
            if test_loss < best_loss:
                best_loss = test_loss
                best_model = model.state_dict()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'Early stopping at epoch {epoch}')
                    break

    if early_stopping and best_model is not None:
        model.load_state_dict(best_model)
        logger.info(f'Loaded best model from epoch {best_epoch}')

    return {
        'model': model,
        'train_acc': train_accuracies[-1],
        'test_acc': test_accuracies[-1],
        'train_loss': train_losses[-1],
        'test_loss': test_losses[-1],
        'best_epoch': best_epoch,
        'architecture': model.identifier,
        'all_train_losses': train_losses,
        'all_test_losses': test_losses,
        'all_train_accuracies': train_accuracies,
        'all_test_accuracies': test_accuracies,
    }


def save_models(results, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'results.csv')

    logger = logging.getLogger(__name__)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Architecture', 'Train Accuracy', 'Test Accuracy', 'Train Loss', 'Test Loss', 'Best Epoch'])

        for key, res in results.items():
            model = res['model']
            model_name = f"conv_{key}"

            # Save PyTorch model
            model_path = os.path.join(save_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved model weights: {model_path}")

            # Save ONNX model
            try:
                dummy_input = torch.randn(1, 1, 28, 28).to(next(model.parameters()).device)
                onnx_path = os.path.join(save_dir, f"{model_name}.onnx")
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output']
                )
                logger.info(f"Saved ONNX model: {onnx_path}")
            except Exception as e:
                logger.warning(f"Failed to export ONNX model for {model_name}: {e}")

            # Save architecture
            arch_path = os.path.join(save_dir, f'{model_name}_architecture.json')
            with open(arch_path, 'w') as arch_file:
                json.dump(res['architecture'], arch_file, indent=2)

            writer.writerow([
                res['architecture'],
                f"{res['train_acc']:.2f}",
                f"{res['test_acc']:.2f}",
                f"{res['train_loss']:.4f}",
                f"{res['test_loss']:.4f}",
                res['best_epoch']
            ])

    logger.info(f"Results saved to {csv_path}")


def main():
    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])

    if DATASET_NAME == "MNIST":
        dataset_cls = datasets.MNIST
    elif DATASET_NAME == "FMNIST":
        dataset_cls = datasets.FashionMNIST
    else:
        raise ValueError(f"Dataset {DATASET_NAME} non supportato")

    trainset = dataset_cls(root='./data', train=True, download=True, transform=transform)
    testset = dataset_cls(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    input_dim = 28
    output_dim = 10

    if DATASET_NAME == "MNIST":
        stride = 1
        padding = 0
        kernel_size = 5
        filters_number = 17
    elif DATASET_NAME == "FMNIST":
        stride = 1
        padding = 0
        kernel_size = 2
        filters_number = 17

    results = {}

    for hidden_dim in conv_hidden_dims:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Training conv model with hidden dim {hidden_dim}")
        logger.info(f"{'=' * 50}")

        model = CustomConvNN(input_dim, output_dim, filters_number, kernel_size, stride, padding, hidden_dim)
        result = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            l1_bool=L1_BOOL,
            early_stopping=EARLY_STOPPING,
            device=device,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            l1_lambda=L1_LAMBDA,
            learning_rate=LEARNING_RATE,
            use_scheduler=USE_SCHEDULER
        )
        results[hidden_dim] = result

        logger.info(f"Completed training for hidden_dim={hidden_dim}")
        logger.info(f"Final Test Accuracy: {result['test_acc']:.2f}%")

    save_models(results)
    logger.info("All models trained and saved successfully!")


if __name__ == "__main__":
    main()
