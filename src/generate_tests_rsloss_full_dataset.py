import os
import sys
import copy
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from lirpa_bound_propagation import get_bounds_dict

# Add parent directory to the system path
current_directory = os.getcwd()
parent_directory = os.path.dirname(os.path.dirname(current_directory))
sys.path.insert(0, parent_directory)
torch.autograd.set_detect_anomaly(True)

from utils.rs_loss_regularizer import interval_arithmetic_fc, _l_relu_stable
from utils.utils import load_yaml_config, write_results_on_csv

DATASET_DIRECTORY = '../datasets'
OUTPUT_FOLDER = r"networks"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEBUG = False
INPUT_DIM = 28
OUTPUT_DIM = 28

kernel_size = 3
stride = 1
padding = 0
NOISE_RS_LOSS = 0.1



class CustomNN(nn.Module):
    def __init__(self, input_dim: int, hidden_layer_dim: int, output_dim: int):
        super(CustomNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layer_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Definiamo il custom regularizer
def calculate_rs_loss_regularizer(model, hidden_layer_dim, lb, ub, normalized):
    params = list(model.parameters())
    W1, b1 = params[0], params[1]

    lb_1, ub_1 = interval_arithmetic_fc(lb, ub, W1, b1)
    rs_loss = _l_relu_stable(lb_1, ub_1)

    n_unstable_nodes = (lb_1 * ub_1 < 0).sum(dim=1).float().mean().item()

    if DEBUG:
        print(f"{hidden_layer_dim=}")

    if normalized:
        rs_loss = rs_loss / hidden_layer_dim
        rs_loss = (rs_loss + 1) / 2
        assert 0 <= rs_loss <= 1, "RS LOSS not in 0, 1 range"

    return rs_loss, n_unstable_nodes

# Definiamo il custom regularizer
def calculate_rs_loss_regularizer_lirpa(arch_shape, lb, ub, normalized):

    hidden_layer_dim = arch_shape[1]

    rs_loss = _l_relu_stable(lb, ub)

    n_unstable_nodes = (lb * ub < 0).sum(dim=1).float().mean().item()

    if DEBUG:
        print(f"{hidden_layer_dim=}")

    if normalized:
        rs_loss = rs_loss / hidden_layer_dim
        rs_loss = (rs_loss + 1) / 2
        assert 0 <= rs_loss <= 1, "RS LOSS not in 0, 1 range"

    return rs_loss, n_unstable_nodes

def train(
        model, device, train_loader, test_loader, optimizer_cls, optimizer_params, criterion_cls,
        num_epochs, hidden_layer_dim, num_classes, rs_loss_regularizer=None, noise=0.1,
        scheduler_lr_cls=None, scheduler_lr_params=None, val_loader=None
):
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    criterion = criterion_cls()
    scheduler = scheduler_lr_cls(optimizer, **scheduler_lr_params) if scheduler_lr_cls else None

    metrics = {
        'train_loss': None, 'train_accuracy': None, 'test_loss': None, 'test_accuracy': None,
        'loss_1_train': None, 'loss_2_train': None, 'loss_1_test': None, 'loss_2_test': None,
        'lambda': None
    }

    def compute_rs_loss(inputs, model, hidden_layer_dim, noise):
        ubs, lbs = inputs + noise, inputs - noise
        return calculate_rs_loss_regularizer(model, hidden_layer_dim, lbs, ubs, normalized=True)

    def compute_rs_loss_lirpa(inputs, model, hidden_layer_dim, noise):
        bounds_dict = get_bounds_dict(model, batch_data=inputs, eps_noise=noise, optimize_bound_args = {'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1, }}, method="IBP", device=device)
        lbs, ubs= bounds_dict['/input']
        return calculate_rs_loss_regularizer_lirpa(model, hidden_layer_dim, lbs, ubs, normalized=True)


    for epoch in range(num_epochs):
        model.train()
        running_train_loss, running_train_loss_1, running_train_loss_2 = 0.0, 0.0, 0.0
        correct_train, total_train = 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            targets_hot_encoded = F.one_hot(targets, num_classes=num_classes).float()
            loss = criterion(outputs, targets_hot_encoded) if isinstance(criterion, nn.MSELoss) else criterion(outputs,
                                                                                                               targets)

            partial_loss_1, partial_loss_2 = loss.item(), 0.0
            if rs_loss_regularizer:
                rs_loss, _ = compute_rs_loss(inputs, model, hidden_layer_dim, noise)
                rs_loss, _ = compute_rs_loss_lirpa(inputs, model, hidden_layer_dim, noise)
                loss += rs_loss_regularizer * rs_loss
                partial_loss_2 = rs_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            running_train_loss_1 += partial_loss_1
            running_train_loss_2 += partial_loss_2

            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        loss_1_train = running_train_loss_1 / len(train_loader)
        loss_2_train = running_train_loss_2 / len(train_loader)

        metrics.update({
            'train_loss': train_loss, 'train_accuracy': train_accuracy,
            'loss_1_train': loss_1_train, 'loss_2_train': loss_2_train
        })

        model.eval()
        running_test_loss, running_test_loss_1, running_test_loss_2 = 0.0, 0.0, 0.0
        correct_test, total_test = 0, 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                targets_hot_encoded = F.one_hot(targets, num_classes=num_classes).float()
                loss = criterion(outputs, targets_hot_encoded) if isinstance(criterion, nn.MSELoss) else criterion(
                    outputs, targets)

                partial_loss_1, partial_loss_2 = loss.item(), 0.0
                if rs_loss_regularizer:
                    rs_loss,  n_unstable_nodes = compute_rs_loss(inputs, model, hidden_layer_dim, noise)
                    loss += rs_loss_regularizer * rs_loss
                    partial_loss_2 = rs_loss.item()
                    metrics['n_unstable_nodes'] = n_unstable_nodes

                running_test_loss += loss.item()
                running_test_loss_1 += partial_loss_1
                running_test_loss_2 += partial_loss_2

                _, predicted = torch.max(outputs.data, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()

        test_loss = running_test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test
        loss_1_test = running_test_loss_1 / len(test_loader)
        loss_2_test = running_test_loss_2 / len(test_loader)

        metrics.update({
            'test_loss': test_loss, 'test_accuracy': test_accuracy,
            'loss_1_test': loss_1_test, 'loss_2_test': loss_2_test,
            'lambda': rs_loss_regularizer
        })

        print(f"Epoch {epoch + 1}:")
        print(f"  Train -> Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.3f}%")
        print(f"  Test  -> Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.3f}%")

    return metrics


def generate_networks(data_dict, dataset_name, hidden_dim, RS_FACTOR, calculate_epochs):

    print(f"{hidden_dim=}")

    # Unpack data_dict
    optimizer_dict = data_dict['optimizer']
    scheduler_lr_dict = data_dict['scheduler_lr']

    # Create optimizer params dict
    opt_params_with_weight_decay = optimizer_dict.copy()
    optimizer_name = opt_params_with_weight_decay['type']
    del opt_params_with_weight_decay['type']

    opt_params = opt_params_with_weight_decay.copy()
    del opt_params['weight_decay']

    # Create scheduler_lr_dict params dict
    scheduler_lr_params = scheduler_lr_dict.copy()
    scheduler_lr_name = scheduler_lr_params['type']
    del scheduler_lr_params['type']

    # Dataset parameters
    train_dim = int(data_dict['data']['train_dim'])
    test_dim = int(data_dict['data']['test_dim'])

    # NN architectures
    input_dim = int(data_dict['data']['input_dim'])
    output_dim = int(data_dict['data']['output_dim'])

    # Training parameters
    train_batch_size = int(data_dict['training']['train_batch_size'])
    test_batch_size = int(data_dict['training']['test_batch_size'])
    validation_batch_size = int(data_dict['training']['validation_batch_size'])
    validation_percentage = float(data_dict['training']['validation_percentage'])
    loss_name = data_dict['training']['loss_name']
    num_epochs = calculate_epochs(hidden_dim)

    # Set the device (use GPU if available, otherwise fallback to CPU)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Dataset loading and transformation
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
        train_set = datasets.MNIST(root=DATASET_DIRECTORY
, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=DATASET_DIRECTORY
, train=False, download=True, transform=transform)
        input_shape = (28, 28)
    elif dataset_name == 'FMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
        train_set = datasets.FashionMNIST(root=DATASET_DIRECTORY
, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=DATASET_DIRECTORY
, train=False, download=True, transform=transform)
        input_shape = (28, 28)
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_set = datasets.CIFAR10(root=DATASET_DIRECTORY
, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=DATASET_DIRECTORY
, train=False, download=True, transform=transform)
        input_shape = (3, 32, 32)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create a subset
    train_subset = Subset(train_set, range(train_dim))
    test_subset = Subset(test_set, range(test_dim))

    if scheduler_lr_name == "ReduceLROnPlateau" and validation_percentage > 0.0:
        # Calculate the size of the validation set
        val_size = int(validation_percentage * len(train_subset))
        train_size = len(train_subset) - val_size

        # Split the training subset into training and validation sets
        train_dataset, val_dataset = random_split(train_subset, [train_size, val_size])
        val_loader = DataLoader(val_dataset, batch_size=validation_batch_size, shuffle=True)
    else:
        val_loader = None
        train_dataset = train_subset

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=test_batch_size, shuffle=False)

    if optimizer_name == 'Adam':
        optimizer_cls = optim.Adam
    elif optimizer_name == 'SGD':
        optimizer_cls = optim.SGD
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Define the loss function
    if loss_name == 'CrossEntropyLoss':
        criterion_cls = nn.CrossEntropyLoss
    elif loss_name == 'MSE':
        criterion_cls = nn.MSELoss
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
    criterion_callable = criterion_cls()

    # Define the scheduler
    scheduler_lr_cls = None
    if scheduler_lr_name == "ReduceLROnPlateau":
        scheduler_lr_cls = optim.lr_scheduler.ReduceLROnPlateau
    else:
        scheduler_lr_params = None

    model = CustomNN(784, hidden_dim, 10)
    # Train the model without batch or L1 regularization
    print("Model Simple")
    model1 = copy.deepcopy(model).to(device)
    metrics1 = train(model1, device, train_loader, test_loader, optimizer_cls, opt_params, criterion_cls, num_epochs,
                     hidden_layer_dim=hidden_dim, num_classes=output_dim, rs_loss_regularizer=RS_FACTOR,
                     scheduler_lr_cls=scheduler_lr_cls, scheduler_lr_params=scheduler_lr_params, val_loader=val_loader, noise=NOISE_RS_LOSS)
    metrics1['h_dim'] = hidden_dim

    return metrics1, model1


def save_model(model, hidden_dim, folder, device):
    # Export the models to ONNX format
    dummy_input = torch.rand(1, 784).to(device)  # Ensure input is on the same device

    # Save the model in ONNX and PyTorch formats
    torch.onnx.export(
        model,
        dummy_input,
        f"{folder}/{hidden_dim}.onnx",
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
    )

    torch.save(model, f"{folder}/{hidden_dim}.pth")


def calculate_epochs(filters_number):
    return 1
    if filters_number <= 100:
        num_epochs = 800
    elif filters_number <= 900:
        num_epochs = 1000
    else:
        num_epochs = 1100
    return num_epochs



def get_min_index_and_value(list_of_best_results):
    best_index, best_tuple = min(
        enumerate(list_of_best_results),
        key=lambda x: x[1][1]['n_unstable_nodes']
    )
    return list_of_best_results[best_index][0], list_of_best_results[best_index][1]



if __name__ == '__main__':
    yaml_file = 'config_one_layered_full_dataset.yaml'
    config = load_yaml_config(yaml_file)

    min_increment = 0
    max_increment = 5
    steps_limit = 10

    # Rs lambda for the smallest network, this values has to increase
    rs_factor = 0.001

    for dataset in config['data']['dataset']:

        hidden_layer_dims = [12, 25, 35, 50, 100, 200, 500, 800]
        hidden_layer_dims = [5000, 8000, 10000, 12000]


        OUTPUT_FOLDER = f"networks/{dataset}"
        CSV_FILE = os.path.join(OUTPUT_FOLDER, "results.csv")

        # First model baseline
        first_model = hidden_layer_dims.pop(0)

        # Best accuracy for the smallest netowork. That value is gonna be the value to improve with over-parametrization
        metrics, model  = generate_networks(config, RS_FACTOR=0, hidden_dim=first_model, calculate_epochs=calculate_epochs, dataset_name=dataset)
        save_model(model, first_model, OUTPUT_FOLDER, device)
        write_results_on_csv(CSV_FILE, metrics)

        # Baseline accuracy
        previous_accuracy = metrics['test_accuracy']
        previous_unstable_nodes = None

        # Dict that collects all data of the networks that have improved the neurons stability of the NN
        best_models_dict = dict()

        for idx, h_dim in enumerate(hidden_layer_dims):
            print(f"Number of h_dim {hidden_layer_dims}")

            target_model = None
            target_metrics = None
            target_rs_loss = None
            old_weights = None

            best_models_dict[str(h_dim)] = list()

            # Increasing the lambda parameter of the RSLoss until it stops decreasing the accuracy
            min_increment = 0.1
            max_increment = 7
            increment = (max_increment - min_increment)/2

            steps_counter = 0
            failure_bool = True
            exception_counter = 0

            while steps_counter <= steps_limit:
                try:
                    metrics, model = generate_networks(config,
                                                       RS_FACTOR=rs_factor + increment,
                                                       hidden_dim=hidden_layer_dims[idx], calculate_epochs=calculate_epochs, dataset_name=dataset)
                except Exception as e:
                    print("Exception: ", e)

                    if exception_counter == 3:
                        print("Experiment aborted: more that 3 exceptions launched")

                    print("Exception: ", e)
                    print("Trying again...")
                    exception_counter += 1
                    continue

                if metrics['test_accuracy'] + 0.006 >= previous_accuracy:
                    target_rs_loss = rs_factor + increment
                    min_increment = increment
                    increment = min_increment + (max_increment - min_increment)/2
                    best_models_dict[str(h_dim)].append((model, metrics))
                    failure_bool = False

                    if previous_unstable_nodes is not None:
                        if metrics['n_unstable_nodes'] < previous_unstable_nodes:
                            break


                elif metrics['test_accuracy'] < previous_accuracy:
                    max_increment = increment
                    increment = min_increment +(max_increment - min_increment)/2

                steps_counter += 1

                if min_increment > max_increment:
                    raise ValueError("min_increment > max_increment. Error in binary research implementation")

            if not failure_bool:
                model, metrics = get_min_index_and_value(best_models_dict[str(h_dim)])

                # Old and new (better accuracy) accuracy
                old_accuracy = previous_accuracy
                new_accuracy = metrics['test_accuracy']

                # Old and new (better accuracy) accuracy
                old_unstable_nodes = previous_unstable_nodes
                new_unstable_nodes = metrics['n_unstable_nodes']


                previous_accuracy = new_accuracy
                previous_unstable_nodes = new_unstable_nodes
                save_model(model, h_dim, OUTPUT_FOLDER, device)
                write_results_on_csv(CSV_FILE, metrics)
                rs_factor = target_rs_loss
                print(f"Accuracy of network with {h_dim} has set the accuracy minimum to {previous_accuracy}")

                print(f"{metrics=}")

            else:
                print(f"Network with {h_dim} filters has failed.")

            print(best_models_dict)