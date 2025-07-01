from typing import Any
import cProfile
import pstats
import os
import sys

from auto_LiRPA import BoundedTensor

# Add parent directory and current directory to the system path
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)
sys.path.insert(0, parent_directory)

from torch import nn, Tensor

from training.utils.logger import setup_logger
import logging

from training.hyper_params_search import BinaryHyperParamsResearch
from training.utils.nn_models import CustomConvNN
from training.one_rs_param.regularized_trainer_DOUBLE_PARAM import ModelTrainingManager
from utils.rs_loss_regularizer import calculate_rs_loss_regularizer_conv, _l_relu_stable_conv, _l_relu_stable

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
        normalized = True

        optimize_bound_args = {
            "enable_beta_crown": False,
            "enable_alpha_crown": False,
        }

        x_perturbed = BoundedTensor(input_batch, perturbation)

        _, _ = model.compute_bounds(x=(x_perturbed,), method=method)
        model.set_bound_opts(optimize_bound_args)

        save_dict = model.save_intermediate()

        lb_conv, ub_conv = save_dict.get('/x')
        lb_conv = lb_conv.view(lb_conv.shape[0], -1)
        ub_conv = ub_conv.view(ub_conv.shape[0], -1)

        lb_fc, ub_fc = save_dict.get('/10')
        lb_fc = lb_fc.view(lb_fc.shape[0], -1)
        ub_fc = ub_fc.view(ub_fc.shape[0], -1)

        rs_loss_conv = (_l_relu_stable_conv(lb_conv, ub_conv))
        rs_loss_fc = _l_relu_stable(lb_fc, ub_fc)
        n_unstable_nodes = ((lb_conv * ub_conv < 0).sum(dim=1).float().mean() +
                            (lb_fc * ub_fc < 0).sum(dim=1).float().mean())

        if normalized:
            rs_loss_conv = rs_loss_conv /  (lb_conv.shape[1])
            rs_loss_conv = (rs_loss_conv + 1) / 2
            rs_loss_fc = rs_loss_fc / (lb_fc.shape[1])
            rs_loss_fc = (rs_loss_fc + 1) / 2
            assert 0 <= rs_loss_conv <= 1, "RS LOSS not in 0, 1 range"
            assert 0 <= rs_loss_fc <= 1, "RS LOSS not in 0, 1 range"


        return rs_loss_conv, rs_loss_fc, n_unstable_nodes



def main():
    import argparse
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
    conv_filters_dim = [4, 8, 12, 24, 32, 48, 64]

    if dataset_name == "FMNIST":
        fc_layers_dim = [90, 110, 130, 150, 170, 190, 210]
    elif dataset_name == "MNIST":
        fc_layers_dim = [70, 90, 110, 130, 150, 170, 190]

    arch_tuple = [
        (input_dim, output_dim, conv_filters_dim[index], kernel_size, stride, padding, fc_layers_dim[index])
        for index, x in enumerate(conv_filters_dim)]

    config_file_path = "../config_one_layered_full_dataset.yaml"
    hyper_params_search = BinaryHyperParamsResearch(CustomConvNN, config_file_path, dataset_name,
                                                    arch_tuple)
    hyper_params_search.sequential_training(min_increment, max_increment, steps_limit, ModelTrainingManagerConv)

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
