"""RS Loss with Memory Optimizations and GradScaler"""
import numpy as np
import torch
from auto_LiRPA import BoundedTensor, PerturbationLpNorm



def _l_relu_stable(lb, ub, norm_constant=1.0):
    """Compute stable ReLU loss with memory optimization"""
    with torch.cuda.amp.autocast():
        loss = -torch.mean(torch.sum(torch.tanh(1.0 + norm_constant * lb * ub), dim=-1))
        if loss < lb.shape[1] * -1 or loss > lb.shape[1]:
            raise Exception("Error in RS Loss, value exceeding the maximum")
        return loss

def _l_relu_stable_conv(lb, ub, norm_constant=1.0):
    """Compute stable ReLU loss for convolutional layers"""
    with torch.cuda.amp.autocast():
        loss = -torch.mean(torch.sum(torch.tanh(1.0 + norm_constant * lb * ub), dim=-1))
        if loss < lb.shape[1] * -1 or loss > lb.shape[1]:
            raise Exception("Error in RS Loss, value exceeding the maximum")
        return loss

def interval_arithmetic_fc(lb, ub, W, b):
    """Compute interval arithmetic for fully connected layers"""
    if len(W.shape) == 2:
        with torch.cuda.amp.autocast():
            lb = lb.view(lb.shape[0], -1)
            ub = ub.view(ub.shape[0], -1)
            W = W.T
            zeros = torch.zeros_like(W)
            W_max = torch.maximum(W, zeros)
            W_min = torch.minimum(W, zeros)
            new_lb = torch.matmul(lb, W_max) + torch.matmul(ub, W_min) + b
            new_ub = torch.matmul(ub, W_max) + torch.matmul(lb, W_min) + b
            return new_lb, new_ub
    else:
        raise NotImplementedError("Only 2D weight matrices are supported")


def calculate_rs_loss_regularizer_fc(model, hidden_layer_dim, input_batch, eps, normalized=True):
    """Calculate RS loss regularizer for fully connected layers"""

    # Calcola lower bound e upper bound per ogni input nel batch
    input_lb = torch.clamp(input_batch - eps, min=0, max=1)
    input_ub = torch.clamp(input_batch + eps, min=0, max=1)

    params = list(model.parameters())
    W1, b1 = params[0], params[1]

    with torch.cuda.amp.autocast():
        # Forward pass con mixed precision
        lb_1, ub_1 = interval_arithmetic_fc(input_lb, input_ub, W1, b1)
        rs_loss = _l_relu_stable(lb_1, ub_1)
        n_unstable_nodes = (lb_1 * ub_1 < 0).sum(dim=1).float().mean().item()

        if normalized:
            rs_loss = rs_loss / hidden_layer_dim
            rs_loss = (rs_loss + 1) / 2
            assert 0 <= rs_loss <= 1, "RS LOSS not in 0, 1 range"

    return rs_loss, n_unstable_nodes

def calculate_rs_loss_regularizer_fc_2_layers(model,  hidden_layer_dim, lb, ub, normalized):
    """Calculate RS loss regularizer for fully connected layers"""

    params = list(model.parameters())
    W1, b1 = params[0], params[1]
    W2, b2 = params[2], params[3]

    with torch.cuda.amp.autocast():
        # Forward pass con mixed precision
        lb_1, ub_1 = interval_arithmetic_fc(lb, ub, W1, b1)
        lb_2, ub_2 = interval_arithmetic_fc(lb_1, ub_1, W2, b2)
        rs_loss_1 = _l_relu_stable(lb_1, ub_1)
        rs_loss_2 = _l_relu_stable(lb_2, ub_2)
        rs_loss = rs_loss_1 + rs_loss_2
        n_unstable_nodes = (lb_1 * ub_1 < 0).sum(dim=1).float().mean().item() + (lb_2 * ub_2 < 0).sum(dim=1).float().mean().item()

        if normalized:
            rs_loss = rs_loss / (hidden_layer_dim * 2)
            rs_loss = (rs_loss + 1) / 2
            assert 0 - 0.1 <= rs_loss <= 1 + 0.1, f"RS LOSS not in 0, 1 range: {rs_loss}, rs_loss1={rs_loss_1}, rs_loss2={rs_loss_2}, hidden_layer_dim={hidden_layer_dim}, lb1={lb_1.shape}, ub1={ub_1.shape}, lb2={lb_2.shape}, ub2={ub_2.shape}"
    return rs_loss, n_unstable_nodes

def calculate_rs_loss_regularizer_resnet18(model, lb, ub, normalized=True):
    """
    Calcola RS loss e numero di neuroni instabili per i due layer fully connected della ResNet-18.
    lb, ub: lower e upper bound del batch pre-fc
    """
    W1, b1 = model.fc1.weight, model.fc1.bias
    W2, b2 = model.fc2.weight, model.fc2.bias

    hidden_dim1 = W1.shape[0]  # numero di neuroni in fc1
    hidden_dim2 = W2.shape[0]  # numero di neuroni in fc2 (di solito num_classes)

    with torch.cuda.amp.autocast():
        # Propagazione intervalli
        lb_1, ub_1 = interval_arithmetic_fc(lb, ub, W1, b1)
        lb_2, ub_2 = interval_arithmetic_fc(lb_1, ub_1, W2, b2)

        # RS loss per ciascun layer
        rs_loss_1 = _l_relu_stable(lb_1, ub_1)
        rs_loss_2 = _l_relu_stable(lb_2, ub_2)
        rs_loss = rs_loss_1 + rs_loss_2

        # Neuroni instabili
        n_unstable_nodes = ((lb_1 * ub_1 < 0).sum(dim=1).float().mean().item() +
                            (lb_2 * ub_2 < 0).sum(dim=1).float().mean().item())

        if normalized:
            # Normalizza separatamente per ciascun layer
            rs_loss = (rs_loss_1 / hidden_dim1 + rs_loss_2 / hidden_dim2) / 2
            rs_loss = (rs_loss + 1) / 2
            assert -0.1 <= rs_loss <= 1.1, f"RS LOSS out of range: {rs_loss}"

    return rs_loss, n_unstable_nodes


    with torch.cuda.amp.autocast():
        # Forward pass con mixed precision
        lb_1, ub_1 = interval_arithmetic_fc(lb, ub, W1, b1)
        lb_2, ub_2 = interval_arithmetic_fc(lb_1, ub_1, W2, b2)
        rs_loss_1 = _l_relu_stable(lb_1, ub_1)
        rs_loss_2 = _l_relu_stable(lb_2, ub_2)
        rs_loss = rs_loss_1 + rs_loss_2
        n_unstable_nodes = (lb_1 * ub_1 < 0).sum(dim=1).float().mean().item() + (lb_2 * ub_2 < 0).sum(dim=1).float().mean().item()

        if normalized:
            rs_loss = rs_loss / (hidden_layer_dim * 2)
            rs_loss = (rs_loss + 1) / 2
            assert 0 - 0.1 <= rs_loss <= 1 + 0.1, f"RS LOSS not in 0, 1 range: {rs_loss}, rs_loss1={rs_loss_1}, rs_loss2={rs_loss_2}, hidden_layer_dim={hidden_layer_dim}, lb1={lb_1.shape}, ub1={ub_1.shape}, lb2={lb_2.shape}, ub2={ub_2.shape}"
    return rs_loss, n_unstable_nodes

def calculate_rs_loss_regularizer_conv(model_lirpa, architecture_tuple, input_batch, perturbation, method, normalized):
    optimize_bound_args = {
        "enable_beta_crown": False,
        "enable_alpha_crown": False,
    }
    x_L = torch.clamp(input_batch[0] - perturbation, min=0, max=1)
    x_U = torch.clamp(input_batch[0] + perturbation, min=0, max=1)
    ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=x_L, x_U=x_U)
    x_perturbed = BoundedTensor(input_batch[0], ptb)


    _, _ = model_lirpa.compute_bounds(x=(x_perturbed,), method=method)
    model_lirpa.set_bound_opts(optimize_bound_args)

    save_dict = model_lirpa.save_intermediate()

    
    lb_conv, ub_conv = save_dict.get('/x')
    lb_conv = lb_conv.view(lb_conv.shape[0], -1)
    ub_conv = ub_conv.view(ub_conv.shape[0], -1)

    lb_fc, ub_fc = save_dict.get('/10')
    lb_fc = lb_fc.view(lb_fc.shape[0], -1)
    ub_fc = ub_fc.view(ub_fc.shape[0], -1)

    rs_loss = _l_relu_stable_conv(lb_conv, ub_conv) + _l_relu_stable(lb_fc, ub_fc)
    n_unstable_nodes = ((lb_conv * ub_conv < 0).sum(dim=1).float().mean() +
                        (lb_fc * ub_fc < 0).sum(dim=1).float().mean())

    if normalized:
        rs_loss = rs_loss / (lb_fc.shape[1] + lb_conv.shape[1])
        rs_loss = (rs_loss + 1) / 2
        assert 0 <= rs_loss <= 1, (f"RS LOSS not in 0, 1 range: rs_loss={rs_loss}, lb_fc.shape={lb_fc.shape}, "
                                   f"lb_conv.shape={lb_conv.shape}, architecture_tuple={architecture_tuple}")

    return rs_loss, n_unstable_nodes


def calculate_symb_bounds(model_lirpa, architecture_tuple, input_batch, perturbation, method="ibp", normalized=True):
    optimize_bound_args = {
        "enable_beta_crown": False,
        "enable_alpha_crown": False,
    }
    # Creating a subset
    sub_inputs, _ = random_subbatch_per_class(input_batch=input_batch[0], labels_batch=input_batch[1], samples_per_class=1)

    perturbation = PerturbationLpNorm(norm=np.inf, eps=perturbation)
    x_perturbed = BoundedTensor(sub_inputs, perturbation)

    _, _ = model_lirpa.compute_bounds(x=(x_perturbed,), method=method)
    model_lirpa.set_bound_opts(optimize_bound_args)
    #print(model_lirpa.named_modules())
    save_dict = model_lirpa.save_intermediate()

    list_of_keys = ['/input', '/input-3', '/input-7', '/input-11']
    n_of_hidden_layer = architecture_tuple[1]
    n_of_neurons = n_of_hidden_layer * architecture_tuple[2]

    n_unstable_nodes = 0
    list_of_bounds = []

    for i in range(n_of_hidden_layer):
        lb, ub = save_dict.get(list_of_keys[i])
        lb = lb.view(lb.shape[0], -1)
        ub = ub.view(ub.shape[0], -1)
        list_of_bounds.append((lb, ub))
        n_unstable_nodes += (lb * ub < 0).sum(dim=1).float().mean().item()

    rs_loss = 0
    for i in range(len(list_of_bounds)):
        rs_loss += _l_relu_stable(list_of_bounds[i][0], list_of_bounds[i][1])

    if normalized:
        rs_loss = rs_loss / n_of_neurons
        rs_loss = (rs_loss + 1) / 2
        assert 0 <= rs_loss <= 1, "RS LOSS not in 0, 1 range"

    return rs_loss, n_unstable_nodes

def random_subbatch_per_class(input_batch, labels_batch, samples_per_class):
    """
    Crea una sottobatch randomica con un certo numero di sample per ogni classe.

    Args:
        input_batch (torch.Tensor): Batch di input (shape: [B, ...]).
        labels_batch (torch.Tensor): Etichette corrispondenti (shape: [B]).
        samples_per_class (int): Numero di sample da includere per ogni classe.

    Returns:
        sub_inputs (torch.Tensor): Sotto-batch di input.
        sub_labels (torch.Tensor): Sotto-batch di etichette.
    """
    selected_indices = []

    classes = labels_batch.unique()
    for cls in classes:
        indices = (labels_batch == cls).nonzero(as_tuple=True)[0]
        if len(indices) >= samples_per_class:
            chosen = indices[torch.randperm(len(indices))[:samples_per_class]]
        else:
            # Se ci sono meno sample di quelli richiesti, prendili tutti
            chosen = indices
        selected_indices.append(chosen)

    selected_indices = torch.cat(selected_indices)
    sub_inputs = input_batch[selected_indices]
    sub_labels = labels_batch[selected_indices]

    return sub_inputs, sub_labels

if __name__ == "__main__":
    # Create test tensors
    batch_size = 2
    hidden_dim = 50

    # Test case 1: All positive bounds
    lb_1 = torch.linspace(0.1, 5.0, hidden_dim).unsqueeze(0)
    ub_1 = torch.linspace(5.1, 10.0, hidden_dim).unsqueeze(0)

    lb_2 = torch.linspace(0.2, 5.1, hidden_dim).unsqueeze(0)
    ub_2 = torch.linspace(5.2, 10.1, hidden_dim).unsqueeze(0)

    # Calculate RS losses for positive case
    rs_loss_1 = _l_relu_stable(lb_1, ub_1)
    rs_loss_2 = _l_relu_stable(lb_2, ub_2)
    total_rs_loss = rs_loss_1 + rs_loss_2

    # Print results for positive case
    print("Test Case 1: All Positive Bounds")
    print(f"RS Loss 1: {rs_loss_1:.4f}")
    print(f"RS Loss 2: {rs_loss_2:.4f}")
    print(f"Total RS Loss: {total_rs_loss:.4f}")
    print(f"Shapes: lb_1: {lb_1.shape}, ub_1: {ub_1.shape}, lb_2: {lb_2.shape}, ub_2: {ub_2.shape}")

    rs_loss = total_rs_loss / (hidden_dim * 2)
    rs_loss = (rs_loss + 1) / 2
    assert 0 <= rs_loss <= 1, "RS LOSS not in 0, 1 range"
    print(f"Normalized RS Loss: {rs_loss:.4f}\n")

    # Test case 2: All negative bounds
    lb_1 = torch.linspace(-5.0, -0.1, hidden_dim).unsqueeze(0)
    ub_1 = torch.linspace(10.0, 5.1, hidden_dim).unsqueeze(0)

    lb_2 = torch.linspace(-5.1, -0.2, hidden_dim).unsqueeze(0)
    ub_2 = torch.linspace(10.1, 5.2, hidden_dim).unsqueeze(0)

    # Calculate RS losses for negative case
    rs_loss_1 = _l_relu_stable(lb_1, ub_1)
    rs_loss_2 = _l_relu_stable(lb_2, ub_2)
    total_rs_loss = rs_loss_1 + rs_loss_2

    # Print results for negative case
    print("Test Case 2: All Negative Bounds")
    print(f"RS Loss 1: {rs_loss_1:.4f}")
    print(f"RS Loss 2: {rs_loss_2:.4f}")
    print(f"Total RS Loss: {total_rs_loss:.4f}")
    print(f"Shapes: lb_1: {lb_1.shape}, ub_1: {ub_1.shape}, lb_2: {lb_2.shape}, ub_2: {ub_2.shape}")

    rs_loss = total_rs_loss / (hidden_dim * 2)
    rs_loss = (rs_loss + 1) / 2
    assert 0 <= rs_loss <= 1, "RS LOSS not in 0, 1 range"
    print(f"Normalized RS Loss: {rs_loss:.4f}")
