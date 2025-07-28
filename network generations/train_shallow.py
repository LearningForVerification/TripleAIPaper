import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import csv
import json

from torchvision import datasets
from torchvision.transforms import transforms


MAX_EPOCHS = 20

hidden_layers_dim = [30, 50, 100, 200, 500, 1000, 2000, 4000, 8000]

# class CustomFCNN(nn.Module):
#     """A simple fully connected neural network with one hidden layer.
#
#     Args:
#         input_dim (int): Number of input features
#         hidden_layer_dim (int): Number of neurons in hidden layer
#         output_dim (int): Number of output features
#     """
#
#     def __init__(self, input_dim: int, hidden_layer_dim: int, output_dim: int):
#         """Initialize the CustomFCNN model.
#
#         Args:
#             input_dim (int): Number of input features
#             hidden_layer_dim (int): Number of neurons in hidden layer
#             output_dim (int): Number of output features
#
#         Raises:
#             ValueError: If any of the dimensions are not positive integers
#         """
#         super(CustomFCNN, self).__init__()
#
#         # Input validation
#         if not all(isinstance(x, int) and x > 0 for x in [input_dim, hidden_layer_dim, output_dim]):
#             raise ValueError("All dimensions must be positive integers")
#
#         self.identifier = f"{hidden_layer_dim}"
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(input_dim, hidden_layer_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_layer_dim, output_dim)
#
#         # Save architecture parameters
#         self.architecture = {
#             'input_dim': input_dim,
#             'hidden_layer_dim': hidden_layer_dim,
#             'output_dim': output_dim
#         }
#
#     def forward(self, x):
#         """Forward pass of the network.
#
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
#
#         Returns:
#             torch.Tensor: Output tensor of shape (batch_size, output_dim)
#         """
#         x = self.flatten(x)
#         if x.dim() != 2 or x.size(1) != self.fc1.in_features:
#             raise ValueError(f"Expected input shape (batch_size, {self.fc1.in_features}), got {x.shape}")
#
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x


class CustomFCNN(nn.Module):
    def __init__(self, input_dim: int, hidden_layer_dim: int, output_dim: int, dropout_prob: float = 0.3):
        super(CustomFCNN, self).__init__()

        if not all(isinstance(x, int) and x > 0 for x in [input_dim, hidden_layer_dim, output_dim]):
            raise ValueError("All dimensions must be positive integers")

        self.identifier = f"{hidden_layer_dim}_dropout_{dropout_prob}"
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_layer_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)  # 🔸 Dropout "forte"
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)

        self.architecture = {
            'input_dim': input_dim,
            'hidden_layer_dim': hidden_layer_dim,
            'output_dim': output_dim,
            'dropout_prob': dropout_prob
        }

    def forward(self, x):
        x = self.flatten(x)
        if x.dim() != 2 or x.size(1) != self.fc1.in_features:
            raise ValueError(f"Expected input shape (batch_size, {self.fc1.in_features}), got {x.shape}")
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # 🔸 Applichiamo dropout
        x = self.fc2(x)
        return x

def interval_arithmetic_fc(lb, ub, W, b):
    """Compute interval arithmetic for fully connected layers"""
    if len(W.shape) == 2:
        with torch.cuda.amp.autocast():
            lb = lb.view(lb.shape[0], -1)
            ub = ub.view(ub.shape[0], -1)
            W = W.T
            zeros = torch.zeros_like(W)
            W_max = torch.maximum(W, zeros)
            W_min = torch.minimum(W, zeros)
            new_lb = torch.matmul(lb, W_max) + torch.matmul(ub, W_min) + b
            new_ub = torch.matmul(ub, W_max) + torch.matmul(lb, W_min) + b
            return new_lb, new_ub
    else:
        raise NotImplementedError("Only 2D weight matrices are supported")


def _l_relu_stable(lb, ub, norm_constant=1.0):
    """Compute stable ReLU loss with memory optimization"""
    with torch.cuda.amp.autocast():
        loss = -torch.mean(torch.sum(torch.tanh(1.0 + norm_constant * lb * ub), dim=-1))
        if loss < lb.shape[1] * -1 or loss > lb.shape[1]:
            raise Exception("Error in RS Loss, value exceeding the maximum")
        return loss


def train_model(model, train_loader, test_loader, l1_bool, early_stopping, device=None,
                max_epochs=MAX_EPOCHS, patience=5, l1_lambda=0.001, learning_rate=0.001, use_scheduler=True):

    # Usa CUDA se disponibile
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

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

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
        test_loss = 0
        correct = 0
        total = 0

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

        if scheduler is not None:
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
                    print(f'Early stopping at epoch {epoch}')
                    break

    if early_stopping and best_model is not None:
        model.load_state_dict(best_model)

    # Calcolo dei neuroni instabili su una batch di test
    unstable_nodes = None
    model.eval()
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            unstable_nodes = calculate_rs_loss_regularizer_fc(model, inputs, eps=0.03)
            break  # solo una batch

    return {
        'model': model,
        'train_acc': train_accuracies[-1],
        'test_acc': test_accuracies[-1],
        'train_loss': train_losses[-1],
        'test_loss': test_losses[-1],
        'best_epoch': best_epoch,
        'architecture': model.architecture,
        'unstable_nodes': unstable_nodes
    }


def calculate_rs_loss_regularizer_fc(model, input_batch, eps):
    """Calculate RS loss regularizer for fully connected layers

    Args:
        model: modello pytorch con layer fully connected
        input_batch: batch di input (tensor) con valori normalizzati in [0,1]
        eps: valore di epsilon per l'intervallo di perturbazione

    Returns:
        n_unstable_nodes: numero medio di nodi instabili nel batch
    """

    # Calcola lower bound e upper bound per ogni input nel batch
    input_lb = torch.clamp(input_batch - eps, min=0, max=1)
    input_ub = torch.clamp(input_batch + eps, min=0, max=1)

    params = list(model.parameters())
    W1, b1 = params[0], params[1]

    with torch.cuda.amp.autocast():
        # Passaggio forward con intervallo tramite arithmetic interval layer fully connected
        lb_1, ub_1 = interval_arithmetic_fc(input_lb, input_ub, W1, b1)

        # Calcolo regularizer RS (relativo a stabilità ReLU)
        rs_loss = _l_relu_stable(lb_1, ub_1)

        # Conta nodi instabili (lb_1 e ub_1 di segno opposto)
        n_unstable_nodes = (lb_1 * ub_1 < 0).sum(dim=1).float().mean().item()

    return n_unstable_nodes

def save_models(results, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'results.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Architecture', 'Train Accuracy', 'Test Accuracy', 'Train Loss', 'Test Loss', 'Best Epoch', 'Unstable Nodes'])

        for hidden_dim in hidden_layers_dim:
            model_name = f'fcnn_{hidden_dim}'
            model = results[hidden_dim]['model']

            # Save PyTorch model
            torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}.pth'))

            # Save ONNX model
            dummy_input = torch.randn(1, model.fc1.in_features).to(next(model.parameters()).device)
            torch.onnx.export(model, dummy_input, os.path.join(save_dir, f'{model_name}.onnx'))

            # Save architecture configuration
            arch_path = os.path.join(save_dir, f'{model_name}_architecture.json')
            with open(arch_path, 'w') as arch_file:
                json.dump(results[hidden_dim]['architecture'], arch_file)

            # Qui aggiungi unstable nodes
            writer.writerow([
                hidden_dim,
                results[hidden_dim]['train_acc'],
                results[hidden_dim]['test_acc'],
                results[hidden_dim]['train_loss'],
                results[hidden_dim]['test_loss'],
                results[hidden_dim]['best_epoch'],
                results[hidden_dim]['unstable_nodes']
            ])


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor()

    ])

    trainset = datasets.MNIST(root='./data', train=True,
                              download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False,
                             download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)

    # Define dimensions
    input_dim = 784
    output_dim = 10

    results = {}
    print(f"{device=}")

    # Train models for each hidden layer dimension
    for hidden_dim in hidden_layers_dim:
        print(f"\nTraining model with {hidden_dim} hidden neurons")
        model = CustomFCNN(input_dim, hidden_dim, output_dim)
        result = train_model(
                    model,
                    train_loader,
                    test_loader,
                    l1_bool=True,
                    early_stopping=True,
                    device=device,
                    max_epochs=50,
                    patience=5,
                    l1_lambda=0,
                    learning_rate=0.005,
                    use_scheduler=True
                )

        results[hidden_dim] = result

    # Save all models
    save_models(results)


if __name__ == "__main__":
    main()
