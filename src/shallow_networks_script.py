from datetime import time
from typing import Any
import cProfile
import pstats

from torch import nn, Tensor

from src.lirpa_test.logger import  setup_logger
import logging

from src.lirpa_test.hyper_params_search import BinaryHyperParamsResearch
from src.lirpa_test.nn_models import CustomFCNN
from src.lirpa_test.regularized_trainer import ModelTrainingManager
from utils.rs_loss_regularizer import  calculate_rs_loss_regularizer_fc

DEBUG = False
min_increment = 0.1
max_increment = 6
steps_limit = 20

setup_logger()
logger = logging.getLogger(__name__)
logger.info("Applicazione avviata")


class ModelTrainingManagerShallow(ModelTrainingManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_rsloss(self, model: nn.Module, model_ref, architecture_tuple: tuple, input_batch: Tensor,
                   perturbation, eps, method='ibp') -> tuple[Any, Any]:
        # Input perturbed bounds
        input_lb = input_batch - eps
        input_ub = input_batch + eps


        rs_loss, n_unstable_nodes = calculate_rs_loss_regularizer_fc(model_ref, architecture_tuple[1], input_lb, input_ub, normalized=True)

        return rs_loss, n_unstable_nodes



def main():
    hidden_layers_dim = [30, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 10000]
    hidden_layers_dim = [(784, x, 10) for x in hidden_layers_dim]

    config_file_path = "config_one_layered_full_dataset.yaml"
    hyper_params_search = BinaryHyperParamsResearch(CustomFCNN, config_file_path, "MNIST",
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
