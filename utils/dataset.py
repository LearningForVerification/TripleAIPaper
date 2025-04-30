# Dataset loading and transformation
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import os

DATASET_DIRECTORY = '../datasets'


def get_dataset(dataset_name):

    if dataset_name == 'MNIST':
        dummy_input = torch.randn(1, 1, 784)
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(root=DATASET_DIRECTORY
                                   , train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=DATASET_DIRECTORY
                                  , train=False, download=True, transform=transform)
        input_dim = 784
        output_dim = 10

    elif dataset_name == 'FMNIST':
        dummy_input = torch.randn(1, 1, 784)
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.FashionMNIST(root=DATASET_DIRECTORY
                                          , train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=DATASET_DIRECTORY
                                         , train=False, download=True, transform=transform)
        input_dim = 784
        output_dim = 10

    elif dataset_name == 'CIFAR10':
        dummy_input = torch.randn(1, 3, 32, 32)
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize])
        train_set = datasets.CIFAR10(root=DATASET_DIRECTORY
                                     , train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=DATASET_DIRECTORY
                                    , train=False, download=True, transform=transform)
        input_dim = (3,32,32)
        output_dim = 10

    else: raise ValueError(f"Unsupported dataset: {dataset_name}")

    return  train_set, test_set, dummy_input, input_dim, output_dim


def get_data_loader(dataset_name, train_batch_size, test_batch_size, num_workers=None):
    train_set, test_set, dummy_input, input_dim, output_dim = get_dataset(dataset_name)

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


def get_data_loader_testing(dataset_name, train_batch_size, test_batch_size, train_size=5000, test_size=1000,
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