import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os
import csv
import onnx
import json
import logging

# ===========================
# PARAMETRI CONFIGURABILI
# ===========================
MAX_EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.001
PATIENCE = 5
USE_SCHEDULER = False
EARLY_STOPPING = False
L1_BOOL = False
L1_LAMBDA = 0
DATASET_NAME = "FMNIST"  # oppure "FMNIST"

# Parametri modello ottimizzato
N_FILTERS1 = 16
N_FILTERS2 = 15
KERNEL_SIZE = 5
FC_DIM = 300
STRIDE = 1
PADDING = 0
INPUT_DIM = 28
OUTPUT_DIM = 10


# ===========================
# MODELLO 2 LAYER CONV
# ===========================
class TwoLayerConvNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_filters1, n_filters2, kernel_size, stride, padding, fc_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_filters1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(n_filters1, n_filters2, kernel_size=kernel_size, stride=stride, padding=padding)

        conv_output_dim = (input_dim - kernel_size + 2 * padding) // stride + 1  # dopo conv1
        conv_output_dim = (conv_output_dim - kernel_size + 2 * padding) // stride + 1  # dopo conv2

        self.fc1 = nn.Linear(n_filters2 * conv_output_dim * conv_output_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, output_dim)
        self.identifier = f"2conv_{n_filters1}_{n_filters2}_k{kernel_size}_fc{fc_dim}"

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ===========================
# FUNZIONE DI TRAINING
# ===========================
def train_model(model, train_loader, test_loader, l1_bool=False, early_stopping=False, device=None,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, l1_lambda=L1_LAMBDA, learning_rate=LEARNING_RATE,
                use_scheduler=USE_SCHEDULER):
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


# ===========================
# SALVATAGGIO MODELLI
# ===========================
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

            # Salva PyTorch
            model_path = os.path.join(save_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved model weights: {model_path}")

            # Salva ONNX
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

            # Salva architettura JSON
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


# ===========================
# MAIN
# ===========================
def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
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

    results = {}

    # Addestramento singolo modello ottimizzato
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Training 2-layer conv model")
    logger.info(f"{'=' * 50}")

    model = TwoLayerConvNN(INPUT_DIM, OUTPUT_DIM, N_FILTERS1, N_FILTERS2, KERNEL_SIZE, STRIDE, PADDING, FC_DIM)
    result = train_model(model, train_loader, test_loader, L1_BOOL, EARLY_STOPPING, device,
                         MAX_EPOCHS, PATIENCE, L1_LAMBDA, LEARNING_RATE, USE_SCHEDULER)

    results[f"{N_FILTERS1}_{N_FILTERS2}_k{KERNEL_SIZE}_fc{FC_DIM}"] = result

    logger.info(f"Completed training for 2-layer conv model")
    logger.info(f"Final Test Accuracy: {result['test_acc']:.2f}%")

    save_models(results)
    logger.info("All models trained and saved successfully!")


if __name__ == "__main__":
    main()
