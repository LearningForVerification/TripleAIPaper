import argparse
import os
import torch
import numpy as np
from training.utils.dataset import get_data_loader
torch.multiprocessing.set_sharing_strategy('file_system')

def setup_dataset(dataset_name):
    if dataset_name in ["MNIST", "FMNIST"]:
        return get_data_loader(
            dataset_name,
            train_batch_size=1,
            test_batch_size=1,
            input_flattened=True,
            num_workers=4
        )
    raise ValueError("Unsupported dataset")


def get_first_n_samples(dataloader, n):
    samples, labels = [], []
    for data, target in dataloader:
        samples.append(data[0].numpy())
        labels.append(target.item())
        if len(samples) >= n:
            break
    return np.array(samples), np.array(labels)


def generate_local_robustness_property(input_sample, noise_level, correct_label, property_path):
    with open(property_path, 'w') as f:
        f.write("; Declare input variables\n")
        for i in range(input_sample.size):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write("\n; Declare output variables\n")
        for i in range(10):  # output classi
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write("\n; Input bounds\n")
        for i in range(input_sample.size):
            lower = max(0, input_sample.flat[i] - noise_level)
            upper = min(1, input_sample.flat[i] + noise_level)
            f.write(f"(assert (>= X_{i} {lower}))\n")
            f.write(f"(assert (<= X_{i} {upper}))\n")

        f.write("\n; Output constraints\n")
        for i in range(10):
            if i != correct_label:
                f.write(f"(assert (>= Y_{i} Y_{correct_label}))\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--property_folder', type=str, required=True, help='Folder to save .vnnlib files')
    parser.add_argument('--dataset_name', type=str, default='MNIST', choices=['MNIST', 'FMNIST'])
    parser.add_argument('--test_sample', type=int, default=3000, help='Number of samples to generate')
    parser.add_argument('--eps', type=float, default=0.015, help='Epsilon for local robustness')
    parser.add_argument('--prefix', type=str, default='sample', help='Filename prefix')

    args = parser.parse_args()

    os.makedirs(args.property_folder, exist_ok=True)

    # Carica dataset
    _, test_loader, _, _, _ = setup_dataset(args.dataset_name)

    # Prendi primi N sample
    samples, labels = get_first_n_samples(test_loader, args.test_sample)

    # Genera proprietà
    for i, (sample, label) in enumerate(zip(samples, labels)):
        filename = f"{args.prefix}_{i}.vnnlib"
        property_path = os.path.join(args.property_folder, filename)
        generate_local_robustness_property(sample, args.eps, label, property_path)
        print(f"[{i + 1}/{len(samples)}] Salvata: {filename}")
