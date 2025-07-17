import argparse
import os

import numpy as np
from torchvision import datasets
from torchvision.transforms import transforms


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
            value = input_sample.flat[i]
            lower_bound = np.clip(value - noise_level, 0.0, 1.0)
            upper_bound = np.clip(value + noise_level, 0.0, 1.0)

            f.write(f"(assert (>= X_{i} {lower_bound}))\n")
            f.write(f"(assert (<= X_{i} {upper_bound}))\n")

        # Write output constraints ensuring correct classification
        f.write("\n; Output constraints\n")
        for i in range(10):  # Assuming 10 classes for MNIST/FMNIST
            if i != correct_label:
                f.write(f"(assert (>= Y_{i} Y_{correct_label}))\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--property_folder', type=str, required=True, help='Cartella per salvare i file .smt2')
    parser.add_argument('--test_sample', type=int, default=100, help='Numero di campioni da elaborare')
    parser.add_argument('--epsilon', type=float, default=100, help='Forza del rumore')

    args = parser.parse_args()

    # Costanti
    PROPERTY_FOLDER = args.property_folder
    TEST_SAMPLE = args.test_sample
    EPSILON = args.epsilon

    # Creazione cartella se non esiste
    os.makedirs(PROPERTY_FOLDER, exist_ok=True)

    # Caricamento dataset MNIST
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Generazione proprietà SMT
    for idx in range(TEST_SAMPLE):
        input_tensor, label = test_dataset[idx]
        input_np = input_tensor.numpy().squeeze()  # 28x28

        file_name = f"sample_{idx:04d}_label_{label}_eps_{EPSILON:.3f}.vnnlib"
        property_path = os.path.join(PROPERTY_FOLDER, file_name)

        generate_local_robustness_property(input_np, EPSILON, label, property_path)
        print(f"[✓] Salvato: {property_path}")