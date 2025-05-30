import os
import onnx
from onnx import helper


def remove_flatten_nodes(model):
    """Remove flatten nodes from the ONNX model."""
    graph = model.graph
    nodes_to_remove = []
    # Modify input shape  
    graph.input[0].type.tensor_type.shape.ClearField("dim")
    graph.input[0].type.tensor_type.shape.dim.extend([
        helper.make_tensor_value_info("batch_size", onnx.TensorProto.INT64, [1]).type.tensor_type.shape.dim[0],
        helper.make_tensor_value_info("input", onnx.TensorProto.INT64, [784]).type.tensor_type.shape.dim[0],
    ])
    # Remove dimensions 2 and 3 if they exist
    while len(graph.input[0].type.tensor_type.shape.dim) > 2:
        graph.input[0].type.tensor_type.shape.dim.pop()

    # Identify flatten nodes
    for node in graph.node:
        if node.op_type == 'Flatten':
            nodes_to_remove.append(node)

            # Connect input directly to output  
            input_name = node.input[0]
            output_name = node.output[0]

            # Update consumers of the flatten node
            for n in graph.node:
                for i, input_name_consumer in enumerate(n.input):
                    if input_name_consumer == output_name:
                        n.input[i] = input_name

    # Remove flatten nodes
    for node in nodes_to_remove:
        graph.node.remove(node)

    return model


def process_folder(input_folder, output_folder):
    """Process all ONNX files in the input folder."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each .onnx file
    for filename in os.listdir(input_folder):
        if filename.endswith('.onnx'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load model
            model = onnx.load(input_path)

            # Remove flatten nodes
            modified_model = remove_flatten_nodes(model)

            # Save modified model
            onnx.save(modified_model, output_path)


if __name__ == '__main__':
    input_folder = r'C:\Users\andr3\PycharmProjects\TripleAIPaper\network generations\models'
    output_folder = r'C:\Users\andr3\PycharmProjects\TripleAIPaper\network generations\models'
    process_folder(input_folder, output_folder)

    remove_flatten_nodes(onnx.load())