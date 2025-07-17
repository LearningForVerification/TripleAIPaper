import os
import onnx
from onnx import helper

def remove_flatten_nodes(model):
    """Remove Flatten nodes from the ONNX model."""
    graph = model.graph
    nodes_to_remove = []

    # Modifica shape input corretta
    graph.input[0].type.tensor_type.shape.ClearField("dim")
    graph.input[0].type.tensor_type.shape.dim.extend([
        helper.make_tensor_value_info("batch_size", onnx.TensorProto.INT64, [1]).type.tensor_type.shape.dim[0],
        helper.make_tensor_value_info("input", onnx.TensorProto.INT64, [784]).type.tensor_type.shape.dim[0],
    ])
    while len(graph.input[0].type.tensor_type.shape.dim) > 2:
        graph.input[0].type.tensor_type.shape.dim.pop()

    # Rimuovi nodi Flatten e riallinea i collegamenti
    for node in graph.node:
        if node.op_type == 'Flatten':
            nodes_to_remove.append(node)

            input_name = node.input[0]
            output_name = node.output[0]

            for n in graph.node:
                for i, input_name_consumer in enumerate(n.input):
                    if input_name_consumer == output_name:
                        n.input[i] = input_name

    for node in nodes_to_remove:
        graph.node.remove(node)

    return model

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".onnx"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            model = onnx.load(input_path)
            modified_model = remove_flatten_nodes(model)
            onnx.save(modified_model, output_path)
            print(f"Processed: {filename}")

def process_multiple_folders(folder_list):
    for folder in folder_list:
        if not os.path.isdir(folder):
            print(f"Warning: {folder} is not a valid directory, skipping.")
            continue

        flatten_folder = os.path.join(folder, "flatten")
        print(f"Processing folder: {folder}")
        process_folder(folder, flatten_folder)
        print(f"Flattened models saved in: {flatten_folder}\n")

# Esempio di lista cartelle da processare
folder_list = [
    r"/mnt/c/Users/andr3/PycharmProjects/TripleAIPaper/verification_time_testing/networks/2-FC/0.03",
    r"/mnt/c/Users/andr3/PycharmProjects/TripleAIPaper/verification_time_testing/networks/2-FC/not_over_param",
    r"/mnt/c/Users/andr3/PycharmProjects/TripleAIPaper/verification_time_testing/networks/2-FC/over_param",
    r"/mnt/c/Users/andr3/PycharmProjects/TripleAIPaper/verification_time_testing/networks/FC/0.03",
    r"/mnt/c/Users/andr3/PycharmProjects/TripleAIPaper/verification_time_testing/networks/FC/not_over_param",
    r"/mnt/c/Users/andr3/PycharmProjects/TripleAIPaper/verification_time_testing/networks/FC/over_param"
]

process_multiple_folders(folder_list)
