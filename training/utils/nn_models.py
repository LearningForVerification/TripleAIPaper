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

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, last_layer_dim=1048, fc_hidden_dim=256):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.last_layer_dim = last_layer_dim
        self.fc_hidden_dim = fc_hidden_dim

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, last_layer_dim, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Due layer FC con ReLU intermedia
        self.fc1 = nn.Linear(last_layer_dim * block.expansion, fc_hidden_dim)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes)
        self.identifier = f"{last_layer_dim}x{fc_hidden_dim}"

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu_fc(out)
        out = self.fc2(out)
        return out

    def get_shape(self) -> tuple:
        """
        Returns the shape of the network as (input_dim, num_layers, hidden_dim, output_dim).

        Returns:
            tuple: (input_dim, num_layers, hidden_dim, output_dim)
        """
        input_dim = (32, 32)
        num_layers = 2
        hidden_dim = self.fc2.in_features
        output_dim = 10

        return (input_dim, num_layers, hidden_dim, output_dim)

    # ------------------ NUOVO METODO ------------------
    def forward_backbone(self, x):
        """
        Propagazione fino alla fine della backbone, prima dei layer FC.
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

    def estimate_backbone_bounds(self, x0_batch, eps: float, num_samples: int = 50):
        """
        Stima lower e upper bounds della backbone usando campioni Monte Carlo attorno a x0 ± eps.
        """
        self.eval()
        self.to(x0_batch.device)
        all_outputs = []

        with torch.no_grad():
            for x0 in x0_batch:
                x0 = x0.unsqueeze(0)
                # Campioni Monte Carlo uniformi in [x0-eps, x0+eps]
                samples = x0 + (torch.rand(num_samples, *x0.shape[1:], device=x0.device) * 2 - 1) * eps
                samples = torch.clamp(samples, 0.0, 1.0)
                out_samples = self.forward_backbone(samples)
                all_outputs.append(out_samples)

        all_outputs = torch.cat(all_outputs, dim=0)
        lower_bounds = torch.min(all_outputs, dim=0)[0]
        upper_bounds = torch.max(all_outputs, dim=0)[0]

        return lower_bounds, upper_bounds


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


# Classe "solo testa FC"
class FCHeadOnly(nn.Module):
    def __init__(self, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
        super(FCHeadOnly, self).__init__()
        # Creiamo fc1 con gli stessi pesi e bias
        self.fc1 = nn.Linear(fc1_weight.shape[1], fc1_weight.shape[0])
        self.fc1.weight.data.copy_(fc1_weight)
        self.fc1.bias.data.copy_(fc1_bias)

        self.relu = nn.ReLU()

        # Creiamo fc2 con gli stessi pesi e bias
        self.fc2 = nn.Linear(fc2_weight.shape[1], fc2_weight.shape[0])
        self.fc2.weight.data.copy_(fc2_weight)
        self.fc2.bias.data.copy_(fc2_bias)
        self.identifier = f"{fc2_weight.shape[1]}x{fc2_weight.shape[0]}"

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
