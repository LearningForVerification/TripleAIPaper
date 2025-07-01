import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import optuna
import json

# Dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Rete neurale
class ConvNet(nn.Module):
    def __init__(self, n_filters, kernel_size, fc_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, n_filters, kernel_size=kernel_size)
        conv_output_size = 28 - kernel_size + 1
        self.flatten_dim = n_filters * conv_output_size * conv_output_size
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Valutazione

def evaluate(model, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            _, predicted = torch.max(pred, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

# Obiettivo per Optuna

results = []

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_filters = trial.suggest_int("n_filters", 4, 42)
    kernel_size = trial.suggest_int("kernel_size", 2, 5)
    fc_dim = trial.suggest_categorical("fc_dim", [30, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 10000])

    model = ConvNet(n_filters, kernel_size, fc_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(300):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    acc = evaluate(model, device)
    result = {
        "trial": trial.number,
        "n_filters": n_filters,
        "kernel_size": kernel_size,
        "fc_dim": fc_dim,
        "accuracy": acc
    }
    results.append(result)
    print(f"Trial {trial.number}: n_filters={n_filters}, kernel_size={kernel_size}, fc_dim={fc_dim}, acc={acc:.2f}%")
    return acc

# Optuna run
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("\n✅ Miglior configurazione trovata:")
print(study.best_trial)

# Salva i risultati
with open("optuna_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n📁 Risultati salvati in 'optuna_results.json'")
