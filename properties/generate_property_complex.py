import argparse
import re
import os, sys
current_directory = os.path.dirname(os.path.abspath(__file__))  # Gets current file's directory
parent_directory = os.path.dirname(current_directory)           # Gets parent directory
training_path = os.path.join(parent_directory, 'training')      # Creates path to training folder

sys.path.insert(0, parent_directory)
sys.path.insert(0, current_directory)
sys.path.insert(0, training_path)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import onnxruntime as ort
import numpy as np
import os
import subprocess
from os import scandir
from os import path
from training.utils.dataset import get_data_loader
import sys
from jinja2 import Template

# Add parent directory and current directory to the system path
current_directory = os.path.dirname(os.path.abspath(__file__))  # Gets current file's directory
parent_directory = os.path.dirname(current_directory)           # Gets parent directory
training_path = os.path.join(parent_directory, 'training')      # Creates path to training folder

sys.path.insert(0, parent_directory)
sys.path.insert(0, current_directory)
sys.path.insert(0, training_path)

import torch
from torch.utils.data import Dataset, DataLoader
import json
TIMEOUT = 20

class VerifiedDataset(Dataset):
    def __init__(self, samples, labels, max_eps=None, transform=None):
        self.samples = torch.FloatTensor(samples)
        self.labels = torch.LongTensor(labels)
        self.max_eps = max_eps
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        max_eps = self.max_eps[idx] if self.max_eps is not None else None

        if self.transform:
            sample = self.transform(sample)

        return sample, label, max_eps

def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    return session


def setup_dataset():
    if DATASET_NAME == "MNIST":
        train_loader, test_loader, dummy_input, input_dim, output_dim = get_data_loader(
            DATASET_NAME,
            train_batch_size=1,
            test_batch_size=1,
            input_flattened=True,
            num_workers=4
        )
    elif DATASET_NAME == "FMNIST":
        train_loader, test_loader, dummy_input, input_dim, output_dim = get_data_loader(
            DATASET_NAME,
            train_batch_size=1,
            test_batch_size=1,
            input_flattened=True,
            num_workers=4
        )
    else:
        raise ValueError("Unsupported dataset.")
    return train_loader, test_loader, dummy_input, input_dim, output_dim

def get_correctly_classified_samples(model, test_loader, n_samples):
    correct_samples = []
    correct_labels = []
    total_samples = 0
    total_correct = 0

    for data, target in test_loader:
        if len(correct_samples) >= n_samples:
            break

        # Get model prediction
        ort_inputs = {model.get_inputs()[0].name: data.numpy().reshape(1, 1, 28, 28)}
        ort_outputs = model.run(None, ort_inputs)
        predicted = np.argmax(ort_outputs[0], axis=1)

        # Find correctly classified samples
        correct = predicted == target.numpy()
        correct_indices = np.where(correct)[0]

        # Update accuracy metrics
        total_samples += len(target)
        total_correct += len(correct_indices)

        for idx in correct_indices:
            correct_samples.append(data[idx].numpy())
            correct_labels.append(target[idx].item())

    accuracy = total_correct / total_samples
    print(f"Network accuracy on test set: {accuracy:.4f}")
    return np.array(correct_samples), np.array(correct_labels)


def verify_property(model_path, property_file_path):
    """Run alpha-beta-crown verification on generated property file"""

    print(property_file_path)

    # Load and render verification template
    with open(TEMPLATE_PATH) as f:
        template = Template(f.read())

    # Render template with model and property paths
    config = template.render(
        onnx_path=model_path,
        vnnlib_path=property_file_path
    )

    # Save rendered config
    config_path = "config.yaml"
    with open(config_path, 'w') as f:
        f.write(config)

    cmd = [
        sys.executable,
        "complete_verifier/abcrown.py",
        "--config", config_path,
        "--pgd_order", "skip"
    ]
    process = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    if process.returncode != 0:
        print(f"Process output: {process.stdout}")
        print(f"Process error: {process.stderr}")
        raise Exception("Error running abcrown.py")
    # Read the output file
    with open('out.txt', 'r') as f:
        content = f.read()

    # Check verification result
    if 'sat' in content and 'unsat' not in content:
        is_sat = True
    elif 'unsat' in content:
        is_sat = False
    else:
        raise Exception("Error sat/unsat not found")

    return is_sat


def generate_local_robustness_property(input_sample, noise_level, correct_label, property_path):
    """Generate local robustness property for given input and save to file.

    Args:
        input_sample: Input sample (numpy array) 
        noise_level: Maximum allowed perturbation
        correct_label: Correct classification label
        filename: Name of file to save property
    """


    with open(property_path, 'w') as f:
        # Declare input variables 
        f.write("; Declare input variables\n")
        for i in range(input_sample.size):
            f.write(f"(declare-const X_{i} Real)\n")

        # Declare output variables
        f.write("\n; Declare output variables\n")
        for i in range(10):
            f.write(f"(declare-const Y_{i} Real)\n")

        # Write input bounds
        f.write("\n; Input bounds\n")
        for i in range(input_sample.size):
            lower_bound = max(0, input_sample.flat[i] - noise_level)
            upper_bound = min(1, input_sample.flat[i] + noise_level)
            f.write(f"(assert (>= X_{i} {lower_bound}))\n")
            f.write(f"(assert (<= X_{i} {upper_bound}))\n")

        # Write output constraints ensuring correct classification
        f.write("\n; Output constraints\n")
        for i in range(10):  # Assuming 10 classes for MNIST/FMNIST
            if i != correct_label:
                f.write(f"(assert (>= Y_{i} Y_{correct_label}))\n")


def save_verified_dataset(samples_eps_list, correct_samples_list, correct_labels_list, model_path, property_dir, DATASET_NAME):
    # Create directory for reduced dataset
    dataset_dir = os.path.join(property_dir, "verified_dataset")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Collect samples and metadata
    dataset_metadata = {
        "dataset_name": DATASET_NAME,
        "model_name": model_path.split('/')[-1].split('.')[0],
        "total_samples": len(correct_samples_list),
        "samples_info": []
    }

    # Convert to PyTorch tensors
    samples_tensor = torch.FloatTensor(correct_samples_list)
    labels_tensor = torch.LongTensor(correct_labels_list)

    # Collect info for each sample
    for i, (sample, label) in enumerate(zip(correct_samples_list, correct_labels_list)):
        sample_info = {
            "index": i,
            "label": int(label),
            "max_verified_eps": samples_eps_list[i],  # Use saved epsilon for each sample
            "shape": list(sample.shape)
        }
        dataset_metadata["samples_info"].append(sample_info)

    # Save PyTorch dataset
    torch_path = os.path.join(dataset_dir, f"verified_{DATASET_NAME}_dataset.pt")
    torch.save({
        'samples': samples_tensor,
        'labels': labels_tensor,
        'max_verified_eps': samples_eps_list  # Save max eps for each sample
    }, torch_path)

    # Save metadata
    metadata_path = os.path.join(dataset_dir, f"verified_{DATASET_NAME}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(dataset_metadata, f, indent=4)

    print(f"Dataset saved in: {dataset_dir}")
    print(f"Total number of samples: {len(correct_samples_list)}")

    
def load_verified_dataset(dataset_dir, batch_size=32, shuffle=True):
    # Carica il dataset PyTorch
    torch_path = os.path.join(dataset_dir, f"verified_MNIST_dataset.pt")
    data = torch.load(torch_path)
    samples = data['samples']
    labels = data['labels']
    max_eps_dict = data['max_verified_eps']  # Load max eps for each sample

    # Carica i metadati
    metadata_path = os.path.join(dataset_dir, f"verified_MNIST_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Crea il dataset
    dataset = VerifiedDataset(samples, labels)

    # Crea il dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle)
    return dataloader, metadata

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nn_path', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--property_folder', type=str, required=True,
                        help='Folder to store property files')
    parser.add_argument('--template_path', type=str, required=True,
                        help='Path to verification template file')
    parser.add_argument('--refinement_step', type=float, default=0.005, help='Refinement step size')
    parser.add_argument('--dataset_name', type=str, default='MNIST', choices=['MNIST', 'FMNIST'],
                        help='Dataset name')
    parser.add_argument('--test_sample', type=int, default=100, help='Number of test samples')
    parser.add_argument('--min_eps', type=float, default=0.005, help='Minimum epsilon value')
    parser.add_argument('--max_eps', type=float, default=0.05, help='Maximum epsilon value')
    args = parser.parse_args()

    # Set global variables directly
    global NN_PATH, PROPERTY_FOLDER, REFINEMENT_STEP, DATASET_NAME, TEST_SAMPLE, MIN_EPS, MAX_EPS, TEMPLATE_PATH
    NN_PATH = args.nn_path
    PROPERTY_FOLDER = args.property_folder
    TEMPLATE_PATH = args.template_path
    REFINEMENT_STEP = args.refinement_step
    DATASET_NAME = args.dataset_name
    TEST_SAMPLE = args.test_sample
    MIN_EPS = args.min_eps
    MAX_EPS = args.max_eps

    # Load multiple ONNX models from folder
    models_path = []
    model_dir = os.path.abspath(args.nn_path)
    for model_file in os.scandir(model_dir):
        if model_file.is_file() and model_file.name.endswith('.onnx'):
            model_path = os.path.join(model_dir, model_file.name)
            models_path.append(model_path)
            print(f"Loaded model: {model_file.name}")

    property_dir = os.path.abspath(PROPERTY_FOLDER)
    if not os.path.exists(property_dir):
        os.makedirs(property_dir)

    for model_path in models_path:
        print(f"\nProcessing model {model_path}...")
        model_name = os.path.basename(model_path).split('.')[0]
        model_property_dir = os.path.join(property_dir, model_name)

        if not os.path.exists(model_property_dir):
            os.makedirs(model_property_dir)
            print(f"Created directory: {model_property_dir}")

        # Create CSV file for storing results
        csv_path = os.path.join(model_property_dir, "verification_results.csv")
        with open(csv_path, 'w') as f:
            f.write("model,property,verified_epsilon\n")

        model = load_onnx_model(model_path)

        # Setup dataset
        train_loader, test_loader, dummy_input, input_dim, output_dim = setup_dataset()

        # Get correctly classified samples
        correct_samples, correct_labels = get_correctly_classified_samples(model, test_loader, TEST_SAMPLE)

        # List to store max verified epsilon for each sample that is a adversarial example for low eps
        adv_eps_list = []

        # List to store the samples that are adversarial examples for low eps
        adv_samples_list = []

        # List to store the correct labels of the samples that are adversarial examples for low eps
        adv_labels_list = []

        # Generate property for each sample
        for i, (sample, label) in enumerate(zip(correct_samples, correct_labels)):
            filename = f"{model_name}_{i}.vnnlib"
            property_path = os.path.join(model_property_dir, filename)
            generate_local_robustness_property(sample, MIN_EPS, label, property_path)

            try:
                is_sat = verify_property(model_path, property_path)
            except Exception as e:
                print(f"Error during property verification: {str(e)}")
                success = False
                break

            if  is_sat:
                if os.path.exists(property_path):
                    os.remove(property_path)
                continue
            print(f"Property {model_name}_{i} verification result: {'verified' if not is_sat else 'not verified'}")

            max_verified_eps = MIN_EPS
            current_eps = MIN_EPS + REFINEMENT_STEP

            while current_eps <= MAX_EPS:
                print(f"\nTesting epsilon={current_eps:.4f}")
                # Overwrite property file with current epsilon
                generate_local_robustness_property(sample, current_eps, label, property_path)
                try:
                    is_sat = verify_property(model_path, property_path)
                except Exception as e:
                    print(f"Error during property verification: {str(e)}")
                    break

                if current_eps + REFINEMENT_STEP > MAX_EPS:
                    if not is_sat:
                        # If network is safe at max epsilon, delete property file and exit without saving to CSV
                        if os.path.exists(property_path):
                            os.remove(property_path)
                        break
                if is_sat:
                    print(f"Property violated at epsilon={current_eps:.4f}")
                    # Backtrack to last verified epsilon
                    prev_eps = current_eps - REFINEMENT_STEP
                    print(f"Backtracking to epsilon={prev_eps:.4f}")
                    generate_local_robustness_property(sample, prev_eps, label, property_path)
                    print(f"Final verified epsilon={prev_eps:.4f}")
                    max_verified_eps = prev_eps

                    # Store max verified epsilon for this sample
                    adv_eps_list.append(max_verified_eps)
                    adv_samples_list.append(sample)
                    adv_labels_list.append(label)

                    # Write result to CSV only if property was not safe at MAX_EPS
                    with open(csv_path, 'a') as f:
                            f.write(f"{model_name},{filename},{max_verified_eps}\n")
                    break

                max_verified_eps = current_eps
                current_eps += REFINEMENT_STEP


        # Save dataset containing only samples that passed verification in model specific folder
        if len(adv_eps_list) > 0:
            save_verified_dataset(adv_eps_list, adv_samples_list, adv_labels_list, model_path, model_property_dir, DATASET_NAME)


