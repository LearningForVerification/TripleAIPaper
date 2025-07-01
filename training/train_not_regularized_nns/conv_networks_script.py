from typing import Any
import cProfile
import pstats
import os
import sys

# Add parent directory and current directory to the system path
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)
sys.path.insert(0, parent_directory)

from torch import nn, Tensor
import argparse
from training.utils.logger import setup_logger
import logging

from training.one_rs_param.hyper_params_search import BinaryHyperParamsResearch
from training.utils.nn_models import CustomConvNN
from training.one_rs_param.regularized_trainer import ModelTrainingManager
from ..utils.rs_loss_regularizer import calculate_rs_loss_regularizer_conv

STATS = False
min_increment = 0.1
max_increment = 7
steps_limit = 10

setup_logger()
logger = logging.getLogger(__name__)
logger.info("Applicazione avviata")


class ModelTrainingManagerConv(ModelTrainingManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_rsloss(self, model: nn.Module, model_ref, architecture_tuple: tuple, input_batch: Tensor,
                   perturbation, eps: float, method='ibp') -> tuple[Any, Any]:
        rs_loss, n_unstable_nodes = calculate_rs_loss_regularizer_conv(model, architecture_tuple, input_batch,
                                                                       perturbation, method=method, normalized=True)

        return rs_loss, n_unstable_nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, choices=['FMNIST', 'MNIST'],
                        help='Dataset name (FMNIST or MNIST)')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    input_dim = 28
    output_dim = 10
    kernel_size = 3
    stride = 1
    padding = 0
    conv_filters_dim = [8, 16, 24, 32, 48, 64, 96]
    
    if dataset_name == "FMNIST":
        fc_layers_dim = [90, 110, 130, 150, 170, 190, 210]
    elif dataset_name == "MNIST":
        fc_layers_dim = [70, 90, 110, 130, 150, 170, 190]

    arch_tuple = [(input_dim, output_dim, conv_filters_dim[index], kernel_size, stride, padding, fc_layers_dim[index])
                  for index, x in enumerate(conv_filters_dim)]

    config_file_path = "config_one_layered_full_dataset.yaml"
    hyper_params_search = BinaryHyperParamsResearch(CustomConvNN, config_file_path, dataset_name,
                                                    arch_tuple)
    hyper_params_search.binary_search(min_increment, max_increment, steps_limit, ModelTrainingManagerConv)


if __name__ == "__main__":
    setup_logger()

    if STATS:
        profiler = cProfile.Profile()
        profiler.enable()

        main()

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats()

    else:
        main()
