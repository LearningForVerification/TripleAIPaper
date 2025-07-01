from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomFCNN_Shallow(nn.Module):
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
        super(CustomFCNN_Shallow, self).__init__()

        # Input validation
        if not all(isinstance(x, int) and x > 0 for x in [input_dim, hidden_layer_dim, output_dim]):
            raise ValueError("All dimensions must be positive integers")
        
        self.identifier = f"{hidden_layer_dim}"
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_layer_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)

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

    def get_shape(self) -> tuple:
        """
        Returns the shape of the network as (input_dim, hidden_dim, output_dim).
        
        Returns:
            tuple: A tuple containing (input_dim, hidden_dim, output_dim)
        """
        return (self.fc1.in_features, self.fc1.out_features, self.fc2.out_features)

class CustomConvNN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            filters_number: int,
            kernel_size: int,
            stride: int,
            padding: int,
            hidden_layer_dim: int
    ) -> None:
        """
        Custom Convolutional Neural Network implementation.
        
        Args:
            input_dim: Input dimension (square input assumed)
            output_dim: Output dimension
            filters_number: Number of convolutional filters
            kernel_size: Size of convolutional kernel
            stride: Stride for convolution
            padding: Padding for convolution
            hidden_layer_dim: Number of neurons in hidden layer
        """
        super(CustomConvNN, self).__init__()

        if input_dim <= 0 or output_dim <= 0 or filters_number <= 0 or hidden_layer_dim <= 0:
            raise ValueError("Dimensions must be positive")
        if kernel_size > input_dim:
            raise ValueError("Kernel size cannot be larger than input dimension")
        
        self.identifier = f"{filters_number}_{hidden_layer_dim}"

        self.conv = nn.Conv2d(1, filters_number, kernel_size=kernel_size, stride=stride, padding=padding)
        self.flatten = nn.Flatten()

        # Calculate output features mathematically
        conv_output_size = ((input_dim + 2 * padding - kernel_size) // stride + 1)
        fc1_in_features = filters_number * conv_output_size * conv_output_size

        self.fc1 = nn.Linear(fc1_in_features, hidden_layer_dim)
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_shape(self) -> tuple:
        """
        Returns the shapes of the network layers.
        
        Returns:
            tuple: A tuple containing:
                - input shape (assumed square input)
                - conv layer output shape (filters, conv output size, conv output size)
                - fc1 shape (input features, output features) 
                - fc2 shape (input features, output features)
        """
        conv_out_size = ((self.conv.kernel_size[0] + 2 * self.conv.padding[0] - 1) // self.conv.stride[0] + 1)
        return (
            (1, self.conv.kernel_size[0], self.conv.kernel_size[0]),
            (self.conv.out_channels, conv_out_size, conv_out_size),
            (self.fc1.in_features, self.fc1.out_features),
            (self.fc2.in_features, self.fc2.out_features)
        )


class CustomFCNN(nn.Module):
    """A fully connected neural network with configurable number of hidden layers.

    Args:
        input_dim (int): Number of input features
        hidden_layer_dims (tuple): Tuple of the form (num_layers, hidden_dim)
        output_dim (int): Number of output features
    """

    def __init__(self, input_dim: int, hidden_layer_dims: tuple, output_dim: int):
        super(CustomFCNN, self).__init__()

        if not (
                isinstance(input_dim, int) and input_dim > 0 and
                isinstance(output_dim, int) and output_dim > 0 and
                isinstance(hidden_layer_dims, tuple) and len(hidden_layer_dims) == 2
        ):
            raise ValueError(
                "Invalid input: input_dim/output_dim must be positive ints and hidden_layer_dims must be a tuple of (num_layers, hidden_dim)")

        num_layers, hidden_dim = hidden_layer_dims
        if not (isinstance(num_layers, int) and num_layers > 0 and isinstance(hidden_dim, int) and hidden_dim > 0):
            raise ValueError("hidden_layer_dims must contain positive integers (num_layers, hidden_dim)")

        self.identifier = f"{num_layers}x{hidden_dim}"
        self.flatten = nn.Flatten()

        # Build the sequence of layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        if x.dim() != 2 or x.size(1) != self.hidden_layers[0].in_features:
            raise ValueError(
                f"Expected input shape (batch_size, {self.hidden_layers[0].in_features}), got {x.shape}")

        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

    def get_shape(self) -> tuple:
        """
        Returns the shape of the network as (input_dim, num_layers, hidden_dim, output_dim).

        Returns:
            tuple: A tuple containing (input_dim, num_layers, hidden_dim, output_dim)
        """
        num_layers = len([layer for layer in self.hidden_layers if isinstance(layer, nn.Linear)])
        return (
            self.hidden_layers[0].in_features,
            num_layers,
            self.hidden_layers[0].out_features,
            self.output_layer.out_features
        )
