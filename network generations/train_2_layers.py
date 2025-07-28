import os
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MAX_EPOCHS = 30


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


def calculate_rs_loss_regularizer_fc_2_layers(model, inputs, eps=0.03):
    lb = torch.clamp(inputs - eps, min=0, max=1)
    ub = torch.clamp(inputs + eps, min=0, max=1)
    params = list(model.parameters())
    W1, b1 = params[0], params[1]
    W2, b2 = params[2], params[3]
    with torch.cuda.amp.autocast():
        lb_1, ub_1 = interval_arithmetic_fc(lb, ub, W1, b1)
        lb_2, ub_2 = interval_arithmetic_fc(lb_1, ub_1, W2, b2)
        n_unstable = (lb_1 * ub_1 < 0).sum(dim=1).float().mean().item() + (lb_2 * ub_2 < 0).sum(dim=1).float().mean().item()
    return n_unstable



# class CustomFCNN(nn.Module):
#     def __init__(self, input_dim, hidden_layer_dims, output_dim):
#         super().__init__()
#         num_layers, hidden_dim = hidden_layer_dims
#         self.identifier = f"{num_layers}x{hidden_dim}"
#         self.flatten = nn.Flatten()
#
#         layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
#         for _ in range(num_layers - 1):
#             layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
#         self.hidden_layers = nn.Sequential(*layers)
#         self.output_layer = nn.Linear(hidden_dim, output_dim)
#         self.architecture = {
#             "input_dim": input_dim,
#             "num_layers": num_layers,
#             "hidden_dim": hidden_dim,
#             "output_dim": output_dim
#         }
#
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.hidden_layers(x)
#         return self.output_layer(x)


class CustomFCNN(nn.Module):
    def __init__(self, input_dim, hidden_layer_dims, output_dim, dropout_prob=0.3):
        super().__init__()
        num_layers, hidden_dim = hidden_layer_dims
        self.identifier = f"{num_layers}x{hidden_dim}"
        self.flatten = nn.Flatten()

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_prob)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_prob)]
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.architecture = {
            "input_dim": input_dim,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "dropout": dropout_prob
        }

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden_layers(x)
        return self.output_layer(x)


def save_models(results, hidden_layers_dim, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Architecture', 'Train Accuracy', 'Test Accuracy', 'Train Loss', 'Test Loss', 'Best Epoch', 'Unstable Nodes'])
        for dim in hidden_layers_dim:
            model_name = f'fcnn_{dim}'
            model = results[dim]['model']
            torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}.pth'))
            dummy_input = torch.randn(1, 784).to(next(model.parameters()).device)
            torch.onnx.export(model, dummy_input, os.path.join(save_dir, f'{model_name}.onnx'))
            with open(os.path.join(save_dir, f'{model_name}_architecture.json'), 'w') as arch_file:
                json.dump(results[dim]['architecture'], arch_file)
            writer.writerow([
                dim,
                results[dim]['train_acc'],
                results[dim]['test_acc'],
                results[dim]['train_loss'],
                results[dim]['test_loss'],
                results[dim]['best_epoch'],
                results[dim]['unstable_nodes']
            ])


def train_model(model, train_loader, test_loader, l1_bool, early_stopping, device, max_epochs=MAX_EPOCHS, patience=5, l1_lambda=0.001, learning_rate=0.001, use_scheduler=True):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3) if use_scheduler else None
    best_loss, best_model, patience_counter, best_epoch = float('inf'), None, 0, 0
    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    for epoch in range(max_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if l1_bool:
                l1 = sum(torch.norm(p, 1) for p in model.parameters())
                loss += l1_lambda * l1
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
        train_acc = 100 * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)

        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                test_loss += loss.item()
                correct += out.argmax(1).eq(y).sum().item()
                total += y.size(0)
        test_acc = 100 * correct / total
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(test_acc)
        if scheduler:
            scheduler.step(test_loss)

        if early_stopping:
            if test_loss < best_loss:
                best_loss = test_loss
                best_model = model.state_dict()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    if early_stopping and best_model:
        model.load_state_dict(best_model)

    model.eval()
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            unstable_nodes = calculate_rs_loss_regularizer_fc_2_layers(model, x, eps=0.03)
            break

    return {
        'model': model,
        'train_acc': train_accs[-1],
        'test_acc': test_accs[-1],
        'train_loss': train_losses[-1],
        'test_loss': test_losses[-1],
        'best_epoch': best_epoch,
        'architecture': model.architecture,
        'unstable_nodes': unstable_nodes
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)

    input_dim = 784
    output_dim = 10
    results = {}

    hidden_dims = [50, 100, 250, 500, 1000, 2000]
    layer_configs = [(2, dim) for dim in hidden_dims]

    for cfg in layer_configs:
        print(f"\nTraining model with {cfg[1]} hidden neurons and {cfg[0]} layers")
        model = CustomFCNN(input_dim, cfg, output_dim)
        result = train_model(model, train_loader, test_loader, l1_bool=False, early_stopping=True, device=device, learning_rate=0.005, use_scheduler=True)
        results[cfg] = result

    save_models(results, layer_configs)


if __name__ == '__main__':
    main()
