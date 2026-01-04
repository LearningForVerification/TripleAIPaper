import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import csv
import json
import time
import warnings
from typing import Dict, Optional, Tuple

from torchvision import datasets
from torchvision.transforms import transforms

# =========================
# Configurazioni generali
# =========================
MAX_EPOCHS = 100
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.005
DEFAULT_DROPOUT = 0.3         # Dropout per prevenire overfitting
DEFAULT_EPSILON = 0.03
USE_SCHEDULER = False
EARLY_STOPPING = True
PATIENCE = 10                 # Numero epoche di attesa per early stopping
L1_BOOL = False
CHECKPOINT_INTERVAL = 10      # checkpoint frequenti per sicurezza
DATASET_NAME = "MNIST"        # oppure "FMNIST"
HIDDEN_LAYERS_DIM = [30, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 10000]

# =========================
# Definizione rete FCNN
# =========================
class CustomFCNN(nn.Module):
    """Fully Connected Neural Network con un layer nascosto e dropout."""

    def __init__(self, input_dim: int, hidden_layer_dim: int, output_dim: int, dropout_prob: float = DEFAULT_DROPOUT):
        super(CustomFCNN, self).__init__()

        if not all(isinstance(x, int) and x > 0 for x in [input_dim, hidden_layer_dim, output_dim]):
            raise ValueError("All dimensions must be positive integers")
        if not 0 <= dropout_prob < 1:
            raise ValueError("Dropout probability must be in range [0,1)")

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_layer_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)

        self.architecture = {
            'input_dim': input_dim,
            'hidden_layer_dim': hidden_layer_dim,
            'output_dim': output_dim,
            'dropout_prob': dropout_prob
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# =========================
# Funzioni utility
# =========================
def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        max_epochs: int = MAX_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        l1_regularization: bool = False,
        l1_lambda: float = 0.001,
        early_stopping: bool = True,
        patience: int = PATIENCE,
        use_scheduler: bool = USE_SCHEDULER,
        save_checkpoints: bool = True,
        checkpoint_dir: Optional[str] = None
) -> Dict:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = (optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
                 if use_scheduler else None)

    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    if save_checkpoints and checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if l1_regularization:
                l1_norm = sum(param.abs().sum() for param in model.parameters())
                loss += l1_lambda * l1_norm

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * train_correct / train_total
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_accuracy = 100. * test_correct / test_total
        test_loss /= len(test_loader)

        if scheduler:
            scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1}/{max_epochs} | "
              f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
              f"Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}% | "
              f"Time: {epoch_time:.2f}s")

        # Checkpoint
        if save_checkpoints and checkpoint_dir and (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), path)

        # Early stopping
        if early_stopping:
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

    if early_stopping and best_model_state:
        model.load_state_dict(best_model_state)
        model.to(device)

    return {
        'model': model,
        'train_acc': train_accuracies[-1],
        'test_acc': test_accuracies[-1],
        'train_loss': train_losses[-1],
        'test_loss': test_losses[-1],
        'best_epoch': best_epoch,
        'total_epochs': len(train_losses),
        'architecture': model.architecture,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }

# =========================
# Funzione per salvare modelli
# =========================
def save_models(results: Dict[int, Dict], save_dir: str = 'models'):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'results.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Architecture', 'Train Acc', 'Test Acc', 'Train Loss', 'Test Loss', 'Best Epoch', 'Total Epochs'])
        for hidden_dim, result in results.items():
            writer.writerow([
                hidden_dim,
                f"{result['train_acc']:.2f}",
                f"{result['test_acc']:.2f}",
                f"{result['train_loss']:.4f}",
                f"{result['test_loss']:.4f}",
                result['best_epoch'],
                result['total_epochs']
            ])
            # Salvataggio modello
            model_path = os.path.join(save_dir, f'fcnn_{hidden_dim}.pth')
            torch.save(result['model'].state_dict(), model_path)
    print(f"Results saved in {csv_path}")

# =========================
# Main
# =========================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    transform = transforms.ToTensor()

    if DATASET_NAME == "MNIST":
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif DATASET_NAME == "FMNIST":
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(trainset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(testset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    input_dim = 784
    output_dim = 10

    results = {}

    for hidden_dim in HIDDEN_LAYERS_DIM:
        print(f"\n{'='*50}\nTraining model with {hidden_dim} hidden neurons\n{'='*50}")
        model = CustomFCNN(input_dim, hidden_dim, output_dim, dropout_prob=DEFAULT_DROPOUT)
        checkpoint_dir = os.path.join('checkpoints', f'fcnn_{hidden_dim}')

        try:
            result = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                max_epochs=MAX_EPOCHS,
                learning_rate=DEFAULT_LEARNING_RATE,
                l1_regularization=L1_BOOL,
                early_stopping=EARLY_STOPPING,
                patience=PATIENCE,
                use_scheduler=USE_SCHEDULER,
                save_checkpoints=True,
                checkpoint_dir=checkpoint_dir
            )
            results[hidden_dim] = result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error training model with {hidden_dim} neurons: {e}")
            import traceback
            traceback.print_exc()
            continue

    if results:
        save_models(results)
        print("Training completed. Models saved in 'models/' directory.")

if __name__ == "__main__":
    main()
