# Dataset loading and transformation
import numpy as np
import pandas as pd
import torch
from keras.src.ops import Normalize
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import os

DATASET_DIRECTORY = '../datasets'

class CustomFeatureDataset(Dataset):
    def __init__(self, csv_file, normalize=False):
        df = pd.read_csv(csv_file)

        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)

        if normalize:
            min_vals = X.min(axis=0)
            max_vals = X.max(axis=0)
            print("Normalized!")

            denom = max_vals - min_vals
            denom[denom == 0] = 1.0  # evita divisioni per zero

            X = (X - min_vals) / denom

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataset(dataset_name, input_flattened=True):
    if dataset_name == 'MNIST':
        if input_flattened:
            dummy_input = torch.randn(1, 784)
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.view(-1))])
        else:
            dummy_input = torch.randn(1, 1, 28, 28)
            transform = transforms.Compose([transforms.ToTensor()])

        train_set = datasets.MNIST(root=DATASET_DIRECTORY, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=DATASET_DIRECTORY, train=False, download=True, transform=transform)
        input_dim = 784
        output_dim = 10

    elif dataset_name == 'FMNIST':
        if input_flattened:
            dummy_input = torch.randn(1, 784)
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Lambda(lambda x: x.view(-1))])
        else:
            dummy_input = torch.randn(1, 1, 28, 28)
            transform = transforms.Compose([transforms.ToTensor()])

        train_set = datasets.FashionMNIST(root=DATASET_DIRECTORY, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=DATASET_DIRECTORY, train=False, download=True, transform=transform)
        input_dim = 784
        output_dim = 10

    elif dataset_name == 'CIFAR10':
        dummy_input = torch.randn(1, 3, 32, 32)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = datasets.CIFAR10(root=DATASET_DIRECTORY, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root=DATASET_DIRECTORY, train=False, download=True, transform=transform_test)
        input_dim = (3, 32, 32)
        output_dim = 10

    elif dataset_name == 'CUSTOM_CIFAR10':
        # Qui metti i file CSV salvati in precedenza

        # Percorso della cartella dove si trova lo script
        base_path = os.path.dirname(os.path.abspath(__file__))

        # File CSV relativi al file Python
        #normalizza tra 0 ee 1
        train_csv = os.path.join(base_path, "custom_train.csv")
        test_csv = os.path.join(base_path, "custom_test.csv")

        print(train_csv)
        print(test_csv)

        train_set = CustomFeatureDataset(train_csv, normalize = True)
        test_set = CustomFeatureDataset(test_csv, normalize = True)

        # dummy input a seconda della forma delle features
        sample = pd.read_csv(train_csv).iloc[0, :-1].values
        dummy_input = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        input_dim = sample.shape[0]
        output_dim = 10  # assumendo 10 classi

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_set, test_set, dummy_input, input_dim, output_dim


def get_data_loader(dataset_name, train_batch_size, test_batch_size, input_flattened, num_workers=None):
    train_set, test_set, dummy_input, input_dim, output_dim = get_dataset(dataset_name, input_flattened)

    if num_workers is None:
        num_workers = min(os.cpu_count(), 8)

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, dummy_input, input_dim, output_dim


def get_dataset_testing(dataset_name, train_size=5000, test_size=1000):
    """
    Get dataset with reduced sizes for testing purposes.

    Args:
        dataset_name (str): Name of the dataset ('MNIST', 'FMNIST', or 'CIFAR10')
        train_size (int): Number of samples to use from training set
        test_size (int): Number of samples to use from test set

    Returns:
        tuple: (train_subset, test_subset, dummy_input, input_dim, output_dim)
    """
    train_set, test_set, dummy_input, input_dim, output_dim = get_dataset(dataset_name)

    # Generate indices for random subset selection
    train_indices = torch.randperm(len(train_set))[:train_size]
    test_indices = torch.randperm(len(test_set))[:test_size]

    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_set, train_indices)
    test_subset = torch.utils.data.Subset(test_set, test_indices)

    return train_subset, test_subset, dummy_input, input_dim, output_dim


def get_data_loader_testing(dataset_name, train_batch_size, test_batch_size, train_size=128, test_size=64,
                            num_workers=None):
    """
    Get data loaders with reduced dataset sizes for testing purposes.

    Args:
        dataset_name (str): Name of the dataset ('MNIST', 'FMNIST', or 'CIFAR10')
        train_batch_size (int): Batch size for training data
        test_batch_size (int): Batch size for test data
        train_size (int): Number of samples to use from training set
        test_size (int): Number of samples to use from test set
        num_workers (int, optional): Number of worker processes for data loading

    Returns:
        tuple: (train_loader, test_loader, dummy_input, input_dim, output_dim)
    """
    train_set, test_set, dummy_input, input_dim, output_dim = get_dataset_testing(dataset_name, train_size, test_size)

    if num_workers is None:
        num_workers = min(os.cpu_count(), 8)

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, dummy_input, input_dim, output_dim