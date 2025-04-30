from typing import Any
import torch
from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor
from torch import nn, Tensor


class ModelTrainingManager:
    def __init__(self, model_class, architecture_tuple, device='cpu', **kwargs):
        self.model = model_class(*architecture_tuple).to(device)
        self.device = device
        self.bounded_model = BoundedModule(self.model, torch.empty((1, *architecture_tuple[:1])))

    def get_rsloss(self, model: nn.Module, architecture_tuple: tuple, input_batch: Tensor,
                   perturbation: PerturbationLpNorm, method='ibp') -> tuple[Any, Any]:
        image = BoundedTensor(input_batch, perturbation)
        _, _ = model.compute_bounds(x=(image,), method=method)
        bounds_saved = model.save_intermediate()

        lb, ub = bounds_saved['/input']
        rs_loss, unstable_nodes = self.calculate_rs_loss_regularizer_lirpa(architecture_tuple[1], lb, ub,
                                                                           normalized=True)
        return rs_loss, unstable_nodes

    @staticmethod
    def calculate_rs_loss_regularizer_lirpa(hidden_layer_dim, lb, ub, normalized):
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
