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

from training.utils.logger import  setup_logger
import logging
from training.utils.nn_models import CustomFCNN
from training.train_not_regularized_nns.sequential_search import SequentialTraining
from training.train_not_regularized_nns.trainer import ModelTrainingManager

DEBUG = False
min_increment = 0.1
max_increment = 6
steps_limit = 15

setup_logger()
logger = logging.getLogger(__name__)
logger.info("Applicazione avviata")



def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_name', type=str, required=True, choices=['MNIST', 'FMNIST'],
    #                     help='Dataset name (MNIST or FMNIST)')
    # args = parser.parse_args()


    #dataset_name = args.dataset_name
    dataset_name = "MNIST"

    hidden_layers_dim = [30, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 10000]
    hidden_layer_tuples = [(2, dim) for dim in hidden_layers_dim]
    hidden_layers_dim = [(784, x, 10) for x in hidden_layer_tuples]

    config_file_path = "config_one_layered_full_dataset.yaml"
    hyper_params_search = SequentialTraining(CustomFCNN, config_file_path, dataset_name,
                                             hidden_layers_dim)
    hyper_params_search.sequential_training(ModelTrainingManager)

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
