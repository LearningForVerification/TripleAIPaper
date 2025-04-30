from datetime import time
from typing import Any
import cProfile
import pstats

import torch
from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor
from torch import nn, Tensor
import src.lirpa_test.logger
import logging

from src.generate_tests_rsloss_full_dataset import calculate_rs_loss_regularizer_lirpa
from src.lirpa_test.hyper_params_researcher import BinaryHyperParamsResearch
from src.lirpa_test.nn_models import CustomFCNN
from src.lirpa_test.regularized_trainer import ModelTrainingManager
from utils.rs_loss_regularizer import _compute_bounds_n_layers

min_increment = 0.1
max_increment = 5
steps_limit = 20

logger = logging.getLogger(__name__)
logger.info("Applicazione avviata")


class ModelTrainingManager_Shallow(ModelTrainingManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_rsloss(self, model: nn.Module, model_ref, architecture_tuple: tuple, input_batch: Tensor,
                   perturbation: float, method='ibp') -> tuple[Any, Any, Tensor, Tensor]:
        # Input perturbed bounds
        input_lb = input_batch - perturbation
        input_ub = input_batch + perturbation


        # Get weight and bias of the hidden layer
        weight = model_ref.fc1.weight
        bias = model_ref.fc1.bias

        lb, ub = _compute_bounds_n_layers( input_lb, input_ub, weight, bias)
        rs_loss, n_unstable_nodes = calculate_rs_loss_regularizer_lirpa(architecture_tuple, lb, ub, normalized=True)

        return rs_loss, n_unstable_nodes

    # Definiamo il custom regularizer
    def calculate_rs_loss_regularizer_lirpa(self, hidden_layer_dim, lb, ub, normalized):

        def _l_relu_stable(lb, ub, norm_constant=1.0):
            loss = -torch.mean(torch.sum(torch.tanh(1.0 + norm_constant * lb * ub), dim=-1))

            if loss < lb.shape[1] * -1 or loss > lb.shape[1]:
                raise Exception("Error in RS Loss, value exceeding the maximum")

            return loss

        rs_loss = _l_relu_stable(lb, ub)

        n_unstable_nodes = (lb * ub < 0).sum(dim=1).float().mean().item()


        if normalized:
            rs_loss = rs_loss / hidden_layer_dim
            rs_loss = (rs_loss + 1) / 2
            assert 0 <= rs_loss <= 1, "RS LOSS not in 0, 1 range"

        return rs_loss, n_unstable_nodes


def main():
    hidden_layers_dim = [30, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 10000]
    hidden_layers_dim = [(784, x, 10) for x in hidden_layers_dim]

    config_file_path = "config_one_layered_full_dataset.yaml"
    hyper_params_search = BinaryHyperParamsResearch(CustomFCNN, config_file_path, "MNIST",
                                                    hidden_layers_dim, verbose=True)
    hyper_params_search.binary_search(min_increment, max_increment, steps_limit, ModelTrainingManager_Shallow)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
