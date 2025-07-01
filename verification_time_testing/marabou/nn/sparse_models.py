import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv

# 📦 Dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 🧠 Rete
class CustomFCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Flatten corretto
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 🛠 Funzioni
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            _, predicted = torch.max(pred, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

def train_and_log(arch, device, lambda_l1, global_log, patience=5, max_epochs=20):
    input_dim, hidden_dim, output_dim = arch
    model = CustomFCNN(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_log = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            # ✅ L1 decay
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        param.grad += lambda_l1 * torch.sign(param)

            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total
        test_acc = evaluate(model, test_loader, device)

        print(f"📊 Arch: {hidden_dim}, Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_model_{hidden_dim}.pth")
            best_log = [hidden_dim, epoch + 1, total_loss, train_acc, test_acc]
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹ Early stopping at epoch {epoch+1} for arch {hidden_dim}")
                break

    # Esportazione ONNX
    model.load_state_dict(torch.load(f"best_model_{hidden_dim}.pth"))
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    torch.onnx.export(model, dummy_input, f"sparse_models/model_{hidden_dim}.onnx", input_names=["input"], output_names=["output"], opset_version=11)

    if best_log:
        global_log.append(best_log)

# 🚀 Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lambda_l1 = 1e-5

hidden_layers = [30, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 10000]
architectures = [(784, h, 10) for h in hidden_layers]

# 📈 Log globale solo dei best model
global_log = []

# 🔁 Addestramento
for arch in architectures:
    train_and_log(arch, device, lambda_l1, global_log)

# 💾 Salvataggio log unico (solo best)
with open("all_training_logs.csv", mode="w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Hidden_Dim", "Epoch", "Train_Loss", "Train_Accuracy", "Test_Accuracy"])
    writer.writerows(global_log)
