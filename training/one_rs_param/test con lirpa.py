import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from training.utils.nn_models import CustomFCNN
from training.utils.dataset import get_data_loader
import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


train_data_loader, test_data_loader, dummy_input, input_dim, output_dim = get_data_loader(
    "MNIST", 128, 128, input_flattened=True)

# Create model instance
model = TwoLayerNet(input_dim=784, hidden_dim=30, output_dim=10)
# Wrap model using auto_LiRPA
bounded_model = BoundedModule(model, dummy_input)

# Define perturbation
eps = 0.015
norm = float("inf")
perturbation = PerturbationLpNorm(norm=np.inf, eps=eps)
x = BoundedTensor(next(iter(test_data_loader))[0], perturbation)
# Get bounds
lb, ub = bounded_model.compute_bounds(x=(x,), method="forward")
save_dict = bounded_model.save_intermediate()

print(f"Lower bounds: {lb}")
print(f"Upper bounds: {ub}")
