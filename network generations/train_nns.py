import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import csv
import onnx
import numpy as np
import json

from torchvision import datasets
from torchvision.transforms import transforms

MAX_EPOCHS = 30

hidden_layers_dim = [1000, 30, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 10000]
hidden_layers_dim = [1000]


class CustomFCNN(nn.Module):
    """A simple fully connected neural network with one hidden layer.

    Args:
        input_dim (int): Number of input features 
        hidden_layer_dim (int): Number of neurons in hidden layer
        output_dim (int): Number of output features
    """

    def __init__(self, input_dim: int, hidden_layer_dim: int, output_dim: int):
        """Initialize the CustomFCNN model.

        Args:
            input_dim (int): Number of input features
            hidden_layer_dim (int): Number of neurons in hidden layer 
            output_dim (int): Number of output features

        Raises:
            ValueError: If any of the dimensions are not positive integers
        """
        super(CustomFCNN, self).__init__()

        # Input validation
        if not all(isinstance(x, int) and x > 0 for x in [input_dim, hidden_layer_dim, output_dim]):
            raise ValueError("All dimensions must be positive integers")

        self.identifier = f"{hidden_layer_dim}"
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_layer_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)

        # Save architecture parameters
        self.architecture = {
            'input_dim': input_dim,
            'hidden_layer_dim': hidden_layer_dim,
            'output_dim': output_dim
        }

    def forward(self, x):
        """Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        x = self.flatten(x)
        if x.dim() != 2 or x.size(1) != self.fc1.in_features:
            raise ValueError(f"Expected input shape (batch_size, {self.fc1.in_features}), got {x.shape}")

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, test_loader, device, max_epochs=MAX_EPOCHS, patience=5, l1_lambda=0.001):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

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

            # Add L1 regularization
            l1_reg = torch.tensor(0., requires_grad=True)
            for param in model.parameters():
                l1_reg = l1_reg + torch.norm(param, 1)
            loss = loss + l1_lambda * l1_reg

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * correct / total
        train_loss = train_loss / len(train_loader)

        # Validation phase
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
        test_loss = test_loss / len(test_loader)

        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Early stopping
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

    # Load best model
    model.load_state_dict(best_model)

    return {
        'model': model,
        'train_acc': train_accuracies[-1],
        'test_acc': test_accuracies[-1],
        'train_loss': train_losses[-1],
        'test_loss': test_losses[-1],
        'best_epoch': best_epoch,
        'architecture': model.architecture
    }


def save_models(results, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'results.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Architecture', 'Train Accuracy', 'Test Accuracy', 'Train Loss', 'Test Loss', 'Best Epoch'])

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

            writer.writerow([
                hidden_dim,
                results[hidden_dim]['train_acc'],
                results[hidden_dim]['test_acc'],
                results[hidden_dim]['train_loss'],
                results[hidden_dim]['test_loss'],
                results[hidden_dim]['best_epoch']
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
    input_dim = 784  # 28x28 pixels
    output_dim = 10  # 10 digits

    results = {}

    # Train models for each hidden layer dimension
    for hidden_dim in hidden_layers_dim:
        print(f"\nTraining model with {hidden_dim} hidden neurons")
        model = CustomFCNN(input_dim, hidden_dim, output_dim)
        result = train_model(model, train_loader, test_loader, device)
        results[hidden_dim] = result

    # Save all models
    save_models(results)


if __name__ == "__main__":
    main()
