import copy
import csv
import os

import torch
import yaml
from torch import nn

DEBUG = True

def load_yaml_config(yaml_file_path):
    """
    Carica la configurazione YAML da un file specificato.
    """
    try:
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Errore: Il file {yaml_file_path} non è stato trovato.")
        return None
    except yaml.YAMLError as exc:
        raise Exception(f"Errore nell'apertura del file YAML: {exc}")


def generate_csv(file_path, header: list, verbose=False):
    """
    Creates or overwrites a CSV file with the specified header.

    Args:
        file_path (str): The path to the CSV file.
        header (list): A list of column names for the CSV header.
        verbose (bool): If True, prints status messages.
    """
    if not isinstance(header, list) or not header:
        print("Error: Header must be a non-empty list.")
        return

    try:
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(header)

        if verbose:
            print(f"Successfully updated header for file: {file_path}")

    except Exception as e:
        print(f"Failed to update file {file_path}. Error: {e}")


def write_results_on_csv(file_path, dict_to_write):
    """
    Appends a dictionary of results to a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        dict_to_write (dict): Dictionary containing the data to write.
        header (list): List of column names (CSV header).
        verbose (bool): If True, prints status messages.
    """
    if not isinstance(dict_to_write, dict):
        raise Exception("Error: dict_to_write must be a dictionary.")


    try:
        file_empty = not os.path.exists(file_path) or os.stat(file_path).st_size == 0

        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(dict_to_write.keys()))

            if file_empty:
                writer.writeheader()

            writer.writerow(dict_to_write)


    except Exception as e:
        raise Exception(f"Failed to write to file {file_path}. Error: {e}")


# def combine_batchnorm1d(linear: nn.Linear, batchnorm: nn.BatchNorm1d) -> nn.Linear:
#     """
#     Utility function to combine a BatchNorm1D node with a Linear node in a corresponding Linear node.
#     Parameters
#     ----------
#     linear : Linear
#         Linear to combine.
#     batchnorm : BatchNorm1D
#         BatchNorm1D to combine.
#     Return
#     ----------
#     Linear
#         The Linear resulting from the fusion of the two input nodes.
#
#     """
#
#     l_weight = linear.weight
#     l_bias = linear.bias
#     bn_running_mean = batchnorm.running_mean
#     bn_running_var = batchnorm.running_var
#     bn_weight = batchnorm.weight
#     bn_bias = batchnorm.bias
#     bn_eps = batchnorm.eps
#
#     fused_bias = torch.div(bn_weight, torch.sqrt(bn_running_var + bn_eps))
#     fused_bias = torch.mul(fused_bias, torch.sub(l_bias, bn_running_mean))
#     fused_bias = torch.add(fused_bias, bn_bias)
#
#     fused_weight = torch.diag(torch.div(bn_weight, torch.sqrt(bn_running_var + bn_eps)))
#     fused_weight = torch.matmul(fused_weight, l_weight)
#
#     has_bias = linear.bias is not None
#     fused_linear = nn.Linear(linear.in_features, linear.out_features,
#                               has_bias)
#
#     p_fused_weight = torch.nn.Parameter(fused_weight, requires_grad=False)
#     p_fused_bias = torch.nn.Parameter(fused_bias, requires_grad=False)
#
#     fused_linear.weight = p_fused_weight
#     fused_linear.bias = p_fused_bias
#
#     return fused_linear
#
#
# def combine_batchnorm1d_torch_net(network):
#     """
#     Utilities function to combine all the FullyConnectedNodes followed by BatchNorm1DNodes in corresponding
#     FullyConnectedNodes.
#     Parameters
#     ----------
#     network : SequentialNetwork
#         Sequential Network of interest of which we want to combine the nodes.
#     Return
#     ----------
#     SequentialNetwork
#         Corresponding Sequential Network with the combined nodes.
#
#     """
#     torch_mode = False
#
#     if isinstance(network, networks.NeuralNetwork):
#         py_net = PyTorchConverter().from_neural_network(network)
#         modules = [m for m in py_net.pytorch_network.modules()]
#
#     elif isinstance(network, torch.nn.Module):
#         modules = [m for m in network.modules()]
#         torch_mode = True
#
#     else:
#         raise Exception("Not supported network")
#
#
#     modules = modules[1:]
#     num_modules = len(modules)
#     current_index = 0
#
#     new_modules = []
#
#     while current_index + 1 < num_modules:
#
#         current_node = modules[current_index]
#         next_node = modules[current_index + 1]
#
#         if isinstance(current_node, nn.Linear) and isinstance(next_node, nn.BatchNorm1d):
#             combined_node = combine_batchnorm1d(current_node, next_node)
#             new_modules.append(combined_node)
#             current_index = current_index + 1
#
#         elif isinstance(current_node, nn.Linear):
#             new_modules.append(copy.deepcopy(current_node))
#
#         elif isinstance(current_node, nn.ReLU):
#             new_modules.append(copy.deepcopy(current_node))
#
#         else:
#             raise Exception("Combine Batchnorm supports only ReLU, Linear and BatchNorm1D layers.")
#
#         current_index = current_index + 1
#
#     if not isinstance(modules[current_index], nn.BatchNorm1d):
#         new_modules.append(copy.deepcopy(modules[current_index]))
#
#     if torch_mode:
#         combined_network = nn.Sequential(*new_modules)
#         return combined_network
#     else:
#         temp_pynet = ptl.Sequential(py_net.pytorch_network.identifier, py_net.pytorch_network.input_id, new_modules)
#         combined_pynet = PyTorchNetwork(py_net.identifier, temp_pynet)
#         combined_network = PyTorchConverter().to_neural_network(combined_pynet)
#
#     return combined_network
#
#
# def transfer_weights(src_model, dest_model):
#     src_state_dict = src_model.state_dict()
#     dest_state_dict = dest_model.state_dict()
#
#     keys_src = list(src_state_dict.keys())
#     keys_dest = list(dest_state_dict.keys())
#
#     for index, key in enumerate(keys_src):
#             dest_state_dict[keys_dest[index]] = src_state_dict[key]
#
#     dest_model.load_state_dict(dest_state_dict)
#
#     return dest_model
#
#
# def compare_models(model1, model2, batch_size = 10, input_dim = 784,  device = torch.device("cpu"), tollerance = 1e-3):
#
#     # Simula immagini MNIST casuali
#     random_input = torch.randn(batch_size, input_dim).to(device)
#
#     # Manda i dati in inferenza controllando il tipo di rete
#     if isinstance(model1, networks.SequentialNetwork):
#         model1 = PyTorchConverter().from_neural_network(model1).pytorch_network.to(device)
#     if isinstance(model2, networks.SequentialNetwork):
#         model2 = PyTorchConverter().from_neural_network(model2).pytorch_network.to(device)
#
#     original_output = model1(random_input)
#     converted_output = model2(random_input)
#
#     # Confronta le predizioni
#     diff = torch.abs(original_output - converted_output).max()
#
#     # Controlla se le predizioni sono praticamente uguali
#     if diff > tollerance:
#         raise Exception(f"The two model are different: {diff} > {tollerance}")
#
def save_models(model, identifier, folder, device, dummy_input=None):
    # Export the models to ONNX format
    if dummy_input is  None:
        dummy_input = torch.rand(1, 1, 28, 28).to(device)
    else:
        dummy_input = dummy_input.to(device)

    # Save the model in ONNX and PyTorch formats
    model.eval()
    torch.onnx.export(
        model.to(device),
        dummy_input,
        f"{folder}/{identifier}.onnx",
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
    )

    torch.save(model, f"{folder}/{identifier}.pth")  # Save underlying PyTorch model
