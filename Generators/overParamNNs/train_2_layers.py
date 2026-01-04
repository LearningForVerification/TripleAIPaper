import os
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
MAX_EPOCHS = 1
PATIENCE = 5
LEARNING_RATE = 0.005
L1_LAMBDA = 0
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 128
EPSILON = 0.03
USE_SCHEDULER = False
EARLY_STOPPING = False
L1_BOOL = False
DATASET_NAME = "MNIST"  # oppure "FMNIST"



# ============================================================================
# INTERVAL ARITHMETIC & ROBUSTNESS ANALYSIS
# ============================================================================

def interval_arithmetic_fc(lb, ub, W, b):
    """
    Compute interval arithmetic for fully connected layers.

    Args:
        lb: Lower bound input tensor
        ub: Upper bound input tensor
        W: Weight matrix
        b: Bias vector

    Returns:
        Tuple of (new_lb, new_ub)
    """
    if len(W.shape) != 2:
        raise NotImplementedError("Only 2D weight matrices are supported")

    # Determine if we should use autocast based on device
    use_autocast = lb.device.type == 'cuda'
    
    with torch.cuda.amp.autocast(enabled=use_autocast):
        # Flatten inputs
        lb = lb.view(lb.shape[0], -1)
        ub = ub.view(ub.shape[0], -1)

        # Transpose weight matrix for correct multiplication
        W = W.T

        # Split weights into positive and negative parts
        W_max = torch.clamp(W, min=0)
        W_min = torch.clamp(W, max=0)

        # Compute new bounds
        new_lb = torch.matmul(lb, W_max) + torch.matmul(ub, W_min) + b
        new_ub = torch.matmul(ub, W_max) + torch.matmul(lb, W_min) + b

        return new_lb, new_ub


def calculate_unstable_nodes_2_layers(model, inputs, eps=EPSILON):
    """
    Calculate number of unstable neurons for 2-layer network.

    Args:
        model: Neural network model
        inputs: Input batch
        eps: Perturbation epsilon

    Returns:
        Average number of unstable nodes
    """
    # Create perturbed input bounds
    lb = torch.clamp(inputs - eps, min=0, max=1)
    ub = torch.clamp(inputs + eps, min=0, max=1)

    # Extract model parameters - handle both training mode and eval mode
    params = list(model.parameters())
    
    # For 2-layer network: input->hidden1->output
    # We need to skip dropout layers if present
    if len(params) < 4:
        raise ValueError(f"Expected at least 4 parameters (2 layers x 2 params), got {len(params)}")
    
    W1, b1 = params[0], params[1]
    # Find the output layer parameters (last weight and bias)
    W2, b2 = params[-2], params[-1]

    use_autocast = inputs.device.type == 'cuda'
    
    with torch.cuda.amp.autocast(enabled=use_autocast):
        # First layer bounds
        lb_1, ub_1 = interval_arithmetic_fc(lb, ub, W1, b1)
        
        # Apply ReLU bounds (nodes are unstable if lb < 0 < ub)
        lb_1_relu = torch.clamp(lb_1, min=0)
        ub_1_relu = torch.clamp(ub_1, min=0)

        # Count unstable neurons in first layer
        unstable_layer_1 = (lb_1 * ub_1 < 0).sum(dim=1).float().mean().item()

        # Second layer bounds (output layer - no ReLU)
        lb_2, ub_2 = interval_arithmetic_fc(lb_1_relu, ub_1_relu, W2, b2)

        # Output layer typically doesn't have ReLU, but count unstable nodes for consistency
        unstable_layer_2 = (lb_2 * ub_2 < 0).sum(dim=1).float().mean().item()

        n_unstable = unstable_layer_1 + unstable_layer_2

    return n_unstable


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class CustomFCNN(nn.Module):
    """
    Custom Fully Connected Neural Network with configurable architecture.
    """

    def __init__(self, input_dim, hidden_layer_dims, output_dim, dropout_prob=0.3):
        """
        Args:
            input_dim: Input dimension
            hidden_layer_dims: Tuple of (num_layers, hidden_dim)
            output_dim: Output dimension
            dropout_prob: Dropout probability
        """
        super().__init__()

        num_layers, hidden_dim = hidden_layer_dims
        self.identifier = f"{num_layers}x{hidden_dim}"
        self.flatten = nn.Flatten()
        self.num_layers = num_layers

        # Build hidden layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_prob)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_prob)]

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Store architecture info
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


# ============================================================================
# TRAINING
# ============================================================================

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate model on given data loader.

    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, device,
                max_epochs=MAX_EPOCHS, patience=PATIENCE,
                learning_rate=LEARNING_RATE, l1_lambda=L1_LAMBDA,
                use_l1=False, use_scheduler=True, early_stopping=True):
    """
    Train neural network model.

    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        device: Training device (CPU/GPU)
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        learning_rate: Initial learning rate
        l1_lambda: L1 regularization coefficient
        use_l1: Whether to use L1 regularization
        use_scheduler: Whether to use learning rate scheduler
        early_stopping: Whether to use early stopping

    Returns:
        Dictionary with training results
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )

    best_test_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = 0

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    print(f"Training on device: {device}")
    print(f"Model architecture: {model.identifier}")

    for epoch in range(max_epochs):
        start_time = time.time()

        # ==================== TRAINING ====================
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            # Add L1 regularization if requested
            if use_l1 and l1_lambda > 0:
                l1_norm = sum(torch.norm(p, 1) for p in model.parameters())
                loss += l1_lambda * l1_norm

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total

        # ==================== EVALUATION ====================
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        # Store metrics
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(test_loss)

        # Check for improvement
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Logging
        epoch_time = time.time() - start_time
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f'Epoch {epoch:4d}/{max_epochs} | '
                  f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | '
                  f'Time: {epoch_time:.2f}s')

        # Early stopping check
        if early_stopping and patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Load best model
    if early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    # ==================== CALCULATE UNSTABLE NODES ====================
    model.eval()
    unstable_nodes = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            unstable_nodes += calculate_unstable_nodes_2_layers(model, x, eps=EPSILON)
            num_batches += 1
            if num_batches >= 10:  # Average over first 10 batches for better estimate
                break
    
    unstable_nodes = unstable_nodes / max(num_batches, 1)

    # Use best epoch metrics if early stopping was enabled
    final_train_acc = train_accs[best_epoch] if early_stopping and best_epoch < len(train_accs) else train_accs[-1]
    final_test_acc = test_accs[best_epoch] if early_stopping and best_epoch < len(test_accs) else test_accs[-1]
    final_train_loss = train_losses[best_epoch] if early_stopping and best_epoch < len(train_losses) else train_losses[-1]
    final_test_loss = test_losses[best_epoch] if early_stopping and best_epoch < len(test_losses) else test_losses[-1]

    return {
        'model': model,
        'train_acc': final_train_acc,
        'test_acc': final_test_acc,
        'train_loss': final_train_loss,
        'test_loss': final_test_loss,
        'best_epoch': best_epoch,
        'architecture': model.architecture,
        'unstable_nodes': unstable_nodes,
        'train_history': {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs
        }
    }


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_models(results, layer_configs, save_dir='models'):
    """
    Save trained models and results to disk.

    Args:
        results: Dictionary of training results
        layer_configs: List of layer configurations
        save_dir: Directory to save models
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save CSV results
    csv_path = os.path.join(save_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Architecture', 'Num_Layers', 'Hidden_Dim',
            'Train_Accuracy', 'Test_Accuracy',
            'Train_Loss', 'Test_Loss',
            'Best_Epoch', 'Unstable_Nodes'
        ])

        for cfg in layer_configs:
            num_layers, hidden_dim = cfg
            model_key = cfg

            if model_key not in results:
                continue

            result = results[model_key]
            model = result['model']
            model_name = f'fcnn_{num_layers}layers_{hidden_dim}hidden'

            # Save PyTorch model
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f'{model_name}.pth')
            )

            # Save ONNX model with error handling
            try:
                dummy_input = torch.randn(1, 784).to(next(model.parameters()).device)
                model.eval()  # Ensure model is in eval mode for ONNX export
                torch.onnx.export(
                    model, dummy_input,
                    os.path.join(save_dir, f'{model_name}.onnx'),
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
            except Exception as e:
                print(f"Warning: Failed to export ONNX model for {model_name}: {e}")

            # Save architecture JSON
            with open(os.path.join(save_dir, f'{model_name}_architecture.json'), 'w') as arch_file:
                json.dump(result['architecture'], arch_file, indent=4)

            # Write CSV row
            writer.writerow([
                f'{num_layers}x{hidden_dim}',
                num_layers,
                hidden_dim,
                f"{result['train_acc']:.2f}",
                f"{result['test_acc']:.2f}",
                f"{result['train_loss']:.4f}",
                f"{result['test_loss']:.4f}",
                result['best_epoch'],
                f"{result['unstable_nodes']:.2f}"
            ])

    print(f"\nResults saved to {save_dir}/")
    print(f"Summary written to {csv_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training loop."""
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Dataset preparation
    transform = transforms.ToTensor()
    # Determine which dataset to use based on configuration
    if DATASET_NAME.upper() == "FMNIST":
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=2, pin_memory=True)

    # Model configuration
    input_dim = 784
    output_dim = 10
    hidden_dims = [50, 100, 250, 500, 1000, 2000]
    layer_configs = [(2, dim) for dim in hidden_dims]

    print("\n" + "=" * 70)
    print("TRAINING 2-LAYER NETWORKS")
    print("=" * 70)
    print(f"Configurations: {len(layer_configs)}")
    print(f"Hidden dimensions: {hidden_dims}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Early stopping: {EARLY_STOPPING}")
    print(f"Use scheduler: {USE_SCHEDULER}")
    print("=" * 70 + "\n")

    # Train all configurations
    results = {}
    for i, cfg in enumerate(layer_configs, 1):
        num_layers, hidden_dim = cfg
        print(f"\n[{i}/{len(layer_configs)}] Training: {num_layers} layers × {hidden_dim} neurons")
        print("-" * 70)

        model = CustomFCNN(input_dim, cfg, output_dim, dropout_prob=0.3)

        try:
            result = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                learning_rate=LEARNING_RATE,
                use_l1=L1_BOOL,
                use_scheduler=USE_SCHEDULER,
                early_stopping=EARLY_STOPPING
            )

            results[cfg] = result
            print(f"✓ Completed: Test Acc = {result['test_acc']:.2f}%, Unstable = {result['unstable_nodes']:.2f}")
        
        except Exception as e:
            print(f"✗ Failed to train {cfg}: {e}")
            continue

    # Save all results
    if results:
        save_models(results, layer_configs)
    else:
        print("No models were successfully trained!")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)


if __name__ == '__main__':
    main()