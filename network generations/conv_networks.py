import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import csv
import onnx
import json

MAX_EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.001
PATIENCE = 5
L1_LAMBDA = 0.001
USE_SCHEDULER = False
EARLY_STOPPING = False
L1_BOOL = False
DATASET_NAME = "MNIST"  # oppure "FMNIST"

conv_hidden_dims = [30, 50, 100, 200, 500, 1000]

#
# class CustomConvNN(nn.Module):
#     def __init__(self, input_dim, output_dim, filters_number, kernel_size, stride, padding, hidden_layer_dim):
#         super(CustomConvNN, self).__init__()
#
#         self.identifier = f"{filters_number}_{hidden_layer_dim}"
#
#         self.conv = nn.Conv2d(1, filters_number, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.flatten = nn.Flatten()
#
#         conv_output_size = ((input_dim + 2 * padding - kernel_size) // stride + 1)
#         fc1_in_features = filters_number * conv_output_size * conv_output_size
#
#         self.fc1 = nn.Linear(fc1_in_features, hidden_layer_dim)
#         self.fc2 = nn.Linear(hidden_layer_dim, output_dim)
#
#         self.architecture = {
#             "conv_out_channels": filters_number,
#             "conv_kernel_size": kernel_size,
#             "conv_stride": stride,
#             "conv_padding": padding,
#             "fc1_in_features": fc1_in_features,
#             "fc1_out_features": hidden_layer_dim,
#             "fc2_out_features": output_dim
#         }
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = F.relu(x)
#         x = self.flatten(x)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


class CustomConvNN(nn.Module):
    def __init__(self, input_dim, output_dim, filters_number, kernel_size, stride, padding, hidden_layer_dim, dropout_prob=0.3):
        super(CustomConvNN, self).__init__()

        self.identifier = f"{filters_number}_{hidden_layer_dim}"

        self.conv = nn.Conv2d(1, filters_number, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dropout_conv = nn.Dropout2d(p=dropout_prob)

        conv_output_size = ((input_dim + 2 * padding - kernel_size) // stride + 1)
        fc1_in_features = filters_number * conv_output_size * conv_output_size

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc1_in_features, hidden_layer_dim)
        self.dropout_fc = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)

        self.architecture = {
            "conv_out_channels": filters_number,
            "conv_kernel_size": kernel_size,
            "conv_stride": stride,
            "conv_padding": padding,
            "fc1_in_features": fc1_in_features,
            "fc1_out_features": hidden_layer_dim,
            "fc2_out_features": output_dim,
            "dropout_probability": dropout_prob
        }

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.dropout_conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, test_loader, l1_bool, early_stopping, device=None,
                max_epochs=MAX_EPOCHS, patience=5, l1_lambda=0.001, learning_rate=0.001, use_scheduler=True):

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

    for epoch in range(max_epochs):
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
                    print(f'Early stopping at epoch {epoch}')
                    break

    if early_stopping and best_model is not None:
        model.load_state_dict(best_model)

    return {
        'model': model,
        'train_acc': train_accuracies[-1],
        'test_acc': test_accuracies[-1],
        'train_loss': train_losses[-1],
        'test_loss': test_losses[-1],
        'best_epoch': best_epoch,
        'architecture': model.architecture,
    }


def save_models(results, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'results.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Architecture', 'Train Accuracy', 'Test Accuracy', 'Train Loss', 'Test Loss', 'Best Epoch'])

        for key, res in results.items():
            model = res['model']
            model_name = f"conv_{key}"
            model_path = os.path.join(save_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)

            dummy_input = torch.randn(1, 1, 28, 28).to(next(model.parameters()).device)
            onnx_path = os.path.join(save_dir, f"{model_name}.onnx")
            torch.onnx.export(model, dummy_input, onnx_path)

            arch_path = os.path.join(save_dir, f'{model_name}_architecture.json')
            with open(arch_path, 'w') as arch_file:
                json.dump(res['architecture'], arch_file)

            writer.writerow([
                model.identifier,
                res['train_acc'],
                res['test_acc'],
                res['train_loss'],
                res['test_loss'],
                res['best_epoch']
            ])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    dataset_cls = datasets.MNIST if DATASET_NAME == "MNIST" else datasets.FashionMNIST

    trainset = dataset_cls(root='./data', train=True, download=True, transform=transform)
    testset = dataset_cls(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = 28
    output_dim = 10
    stride = 1
    padding = 0
    kernel_size = 5
    filters_number = 17

    results = {}

    for hidden_dim in conv_hidden_dims:
        print(f"\nTraining conv model with hidden dim {hidden_dim}")
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

    save_models(results)


if __name__ == "__main__":
    main()
