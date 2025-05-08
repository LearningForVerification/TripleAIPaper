"""RS Loss"""
import torch
from auto_LiRPA import BoundedTensor


def _l_relu_stable(lb, ub, norm_constant=1.0):
    loss = -torch.mean(torch.sum(torch.tanh(1.0 + norm_constant * lb * ub), dim=-1))

    if loss < lb.shape[1] * -1 or loss > lb.shape[1]:
        raise Exception("Error in RS Loss, value exceeding the maximum")

    return loss

def _l_relu_stable_conv(lb, ub, norm_constant=1.0):
    loss = -torch.mean(torch.sum(torch.tanh(1.0 + norm_constant * lb * ub), dim=-1))

    if loss < lb.shape[1] * -1 or loss > lb.shape[1]:
        raise Exception("Error in RS Loss, value exceeding the maximum")

    return loss



def interval_arithmetic_fc(lb, ub, W, b):
    # Base case
    if len(W.shape) == 2:
        # Assume le forme Bxm, Bxm, mxn, n
        lb = lb.view(lb.shape[0], -1)
        ub = ub.view(ub.shape[0], -1)
        W = W.T
        W_max = torch.maximum(W, torch.tensor(0.0))
        W_min = torch.minimum(W, torch.tensor(0.0))

        new_lb = torch.matmul(lb, W_max) + torch.matmul(ub, W_min) + b
        new_ub = torch.matmul(ub, W_max) + torch.matmul(lb, W_min) + b
        return new_lb, new_ub

    else:
        raise NotImplementedError


# Definiamo il custom regularizer
def calculate_rs_loss_regularizer_fc(model, hidden_layer_dim, lb, ub, normalized):
    params = list(model.parameters())
    W1, b1 = params[0], params[1]

    lb_1, ub_1 = interval_arithmetic_fc(lb, ub, W1, b1)
    rs_loss = _l_relu_stable(lb_1, ub_1)

    n_unstable_nodes = (lb_1 * ub_1 < 0).sum(dim=1).float().mean().item()

    if normalized:
        rs_loss = rs_loss / hidden_layer_dim
        rs_loss = (rs_loss + 1) / 2
        assert 0 <= rs_loss <= 1, "RS LOSS not in 0, 1 range"

    return rs_loss, n_unstable_nodes


# Definiamo il custom regularizer
def calculate_rs_loss_regularizer_conv(model_lirpa, hidden_layer_dim, input_batch, perturbation, method, normalized):
    # Input con perturbazione
    x_perturbed = BoundedTensor(input_batch, perturbation)

    # Calcolo bound
    _, _ = model_lirpa.compute_bounds(x=(x_perturbed,),
                                      method=method)
    save_dict = model_lirpa.save_intermediate()

    # Getting
    lb_conv, ub_conv = save_dict.get('/x')
    lb_conv, ub_conv = lb_conv.view(lb_conv.shape[0], -1), ub_conv.view(ub_conv.shape[0], -1)

    lb_fc, ub_fc = save_dict.get('/10')
    lb_fc, ub_fc = lb_fc.view(lb_fc.shape[0], -1), ub_fc.view(ub_fc.shape[0], -1)

    rs_loss = _l_relu_stable_conv(lb_conv, ub_conv) + _l_relu_stable(lb_fc, ub_fc)

    n_unstable_nodes = (lb_conv * ub_conv < 0).sum(dim=1).float().mean().item() + (lb_fc * ub_fc < 0).sum(dim=1).float().mean().item()

    if normalized:
        rs_loss = rs_loss / (lb_fc.shape[1]+ lb_conv.shape[1])
        rs_loss = (rs_loss + 1) / 2
        assert 0 <= rs_loss <= 1, "RS LOSS not in 0, 1 range"

    return rs_loss, n_unstable_nodes