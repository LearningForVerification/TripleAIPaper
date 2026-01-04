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

from training.one_rs_param.reset_training.hyper_params_search import BinaryHyperParamsResearch
from training.utils.nn_models import ResNet, BasicBlock
from training.one_rs_param.reset_training.partial_regularized_trainer import ModelTrainingManager
from training.one_rs_param.config import load_config
from training.utils.logger import setup_logger

DEBUG = False
min_increment = 0.1
max_increment = 6
steps_limit = 3

setup_logger()
logger = logging.getLogger(__name__)
logger.info("Applicazione avviata")



def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--file_name', type=str, required=False,
    #                     help='Configuration file name')
    # args = parser.parse_args()

    dataset_name = "CIFAR10"
    config = load_config("config_cifar.ini")
    init_model_path = r"target_model/resnet18_dim2048.pth"

    hidden_layers_dim = [32, 64, 256, 1024]

    # Costruisci il path assoluto del file YAML nella stessa directory dello script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(script_dir, "config_one_layered_full_dataset.yaml")

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"File di configurazione YAML non trovato: {config_file_path}")

    hyper_params_search = BinaryHyperParamsResearch(
        (ResNet, BasicBlock),
        config_file_path,
        config,
        dataset_name,
        hidden_layers_dim
    )

    hyper_params_search.binary_search(min_increment, max_increment, steps_limit, ModelTrainingManager, init_model_path)


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
