import os
import onnx
from onnx import helper

def remove_flatten_nodes(model):
    """Remove Flatten nodes from the ONNX model."""
    graph = model.graph
    nodes_to_remove = []

    # Modify input shape (optional, depends on model requirements)
    graph.input[0].type.tensor_type.shape.ClearField("dim")
    graph.input[0].type.tensor_type.shape.dim.extend([
        helper.make_tensor_value_info("batch_size", onnx.TensorProto.INT64, [1]).type.tensor_type.shape.dim[0],
        helper.make_tensor_value_info("input", onnx.TensorProto.INT64, [784]).type.tensor_type.shape.dim[0],
    ])
    while len(graph.input[0].type.tensor_type.shape.dim) > 2:
        graph.input[0].type.tensor_type.shape.dim.pop()

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

# Esempio d'uso
input_folder = r"C:\Users\andr3\Desktop\primo-giugno-25\MNIST\MNIST-FC\best_models"
output_folder = r"C:\Users\andr3\Desktop\primo-giugno-25\MNIST\MNIST-FC\best_models\flatten"
process_folder(input_folder, output_folder)
