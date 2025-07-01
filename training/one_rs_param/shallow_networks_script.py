import argparse
import os
from datetime import time
from typing import Any
import cProfile
import pstats
import os
import sys

import torch

# Add parent directory and current directory to the system path
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)
sys.path.insert(0, parent_directory)
from torch import nn, Tensor


import logging

from training.one_rs_param.hyper_params_search import BinaryHyperParamsResearch
from training.utils.nn_models import CustomFCNN_Shallow
from training.one_rs_param.regularized_trainer import ModelTrainingManager
from training.one_rs_param.config import load_config
from training.utils.rs_loss_regularizer import  calculate_rs_loss_regularizer_fc
from training.utils.logger import setup_logger

DEBUG = False
min_increment = 0.1
max_increment = 6
steps_limit = 3

setup_logger()
logger = logging.getLogger(__name__)
logger.info("Applicazione avviata")


class ModelTrainingManagerShallow(ModelTrainingManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_rsloss(self, model: nn.Module, model_ref, architecture_tuple: tuple, input_batch: Tensor,
                   perturbation, eps, method='ibp') -> tuple[Any, Any]:
        # Input perturbed bounds
        # Input perturbed bounds con clipping tra 0 e 1

        input_lb = torch.clamp(input_batch[0] - eps, min=0, max=1)
        input_ub = torch.clamp(input_batch[0] + eps, min=0, max=1)

        rs_loss, n_unstable_nodes = calculate_rs_loss_regularizer_fc(model_ref, architecture_tuple[1], input_lb, input_ub, normalized=True)

        return rs_loss, n_unstable_nodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, choices=['MNIST', 'FMNIST'],
                        help='Dataset name (MNIST or FMNIST)')
    parser.add_argument('--file_name', type=str, required=False,
                        help='Configuration file name')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    config = load_config(args.file_name)

    #hidden_layers_dim = [30, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 10000]
    hidden_layers_dim=[50, 500, 4000]
    hidden_layers_dim = [(784, x, 10) for x in hidden_layers_dim]

    config_file_path = "config_one_layered_full_dataset.yaml"
    hyper_params_search = BinaryHyperParamsResearch(CustomFCNN_Shallow, config_file_path, config, dataset_name,
                                                    hidden_layers_dim)
    hyper_params_search.binary_search(min_increment, max_increment, steps_limit, ModelTrainingManagerShallow)

if __name__ == "__main__":
    setup_logger()

    if DEBUG:
        profiler = cProfile.Profile()
        profiler.enable()

        main()

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats()

    else:
        main()
