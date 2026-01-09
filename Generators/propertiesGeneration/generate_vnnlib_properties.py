import os
import argparse
import numpy as np
from torchvision import datasets
from torchvision.transforms import transforms


def generate_local_robustness_property(
        input_sample,
        noise_level,
        correct_label,
        property_path,
        num_classes=10
):
    """
    Genera una proprietà VNNLIB per la robustezza locale di un singolo input.

    Parametri
    ----------
    input_sample : np.ndarray
        L'input della rete (ad esempio un'immagine normalizzata [0,1]).
    noise_level : float
        Massima perturbazione L_infinity ammessa.
    correct_label : int
        Etichetta corretta dell'esempio.
    property_path : str
        Path dove salvare il file .vnnlib generato.
    num_classes : int
        Numero di classi della rete.
    """
    if noise_level < 0:
        raise ValueError(f"noise_level must be non-negative, got {noise_level}")
    if not 0 <= correct_label < num_classes:
        raise ValueError(
            f"correct_label must be in [0, {num_classes - 1}], got {correct_label}"
        )

    os.makedirs(os.path.dirname(property_path), exist_ok=True)
    flat_input = input_sample.reshape(-1)

    with open(property_path, "w") as f:
        # =========================
        # Declare input variables
        # =========================
        f.write("; Declare input variables\n")
        for i in range(flat_input.size):
            f.write(f"(declare-const X_{i} Real)\n")

        # =========================
        # Declare output variables
        # =========================
        f.write("\n; Declare output variables\n")
        for i in range(num_classes):
            f.write(f"(declare-const Y_{i} Real)\n")

        # =========================
        # Input bounds
        # =========================
        f.write("\n; Input bounds (L_infinity perturbation)\n")
        for i, value in enumerate(flat_input):
            value = float(value)
            lower = max(0.0, value - noise_level)
            upper = min(1.0, value + noise_level)
            f.write(f"(assert (>= X_{i} {lower:.10f}))\n")
            f.write(f"(assert (<= X_{i} {upper:.10f}))\n")

        # =========================
        # Output constraints
        # =========================
        f.write("\n; Output constraints\n")
        for i in range(num_classes):
            if i != correct_label:
                f.write(f"(assert (>= Y_{i} Y_{correct_label}))\n")


# =========================
# Argparse
# =========================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate VNNLIB local robustness properties"
    )
    parser.add_argument(
        "--property_folder",
        type=str,
        default="./properties/FMNIST/0.03",
        help="Folder to save .vnnlib files"
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=100,
        help="Number of samples to process"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
        help="L_infinity perturbation bound"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="FMNIST",
        choices=["MNIST", "FMNIST", "CIFAR10"],
        help="Dataset to use"
    )
    return parser.parse_args()


# =========================
# Load dataset
# =========================
def load_dataset(dataset_name):
    transform = transforms.ToTensor()

    if dataset_name == "MNIST":
        return datasets.MNIST("./data", train=False, download=True, transform=transform), 10
    if dataset_name == "FMNIST":
        return datasets.FashionMNIST("./data", train=False, download=True, transform=transform), 10
    if dataset_name == "CIFAR10":
        return datasets.CIFAR10("./data", train=False, download=True, transform=transform), 10

    raise ValueError(f"Unsupported dataset: {dataset_name}")


# =========================
# Main
# =========================
if __name__ == "__main__":
    args = parse_arguments()

    if args.epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {args.epsilon}")
    if args.test_samples <= 0:
        raise ValueError(f"test_samples must be positive, got {args.test_samples}")

    os.makedirs(args.property_folder, exist_ok=True)

    test_dataset, num_classes = load_dataset(args.dataset)

    if args.test_samples > len(test_dataset):
        raise ValueError(
            f"Requested {args.test_samples} samples, but dataset has only {len(test_dataset)}"
        )

    print(f"Generating {args.test_samples} properties")
    print(f"Dataset: {args.dataset}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Output folder: {args.property_folder}")

    success_count = 0

    for idx in range(args.test_samples):
        input_tensor, label = test_dataset[idx]
        input_np = input_tensor.numpy()

        # CIFAR10: CHW -> HWC
        if input_np.ndim == 3 and input_np.shape[0] == 3:
            input_np = np.transpose(input_np, (1, 2, 0))

        property_path = os.path.join(
            args.property_folder,
            f"sample_{idx:04d}_label_{label}_eps_{args.epsilon:.4f}.vnnlib"
        )

        try:
            generate_local_robustness_property(
                input_np,
                args.epsilon,
                label,
                property_path,
                num_classes=num_classes
            )
            success_count += 1
        except Exception as e:
            print(f"[✗] Failed on sample {idx}: {e}")

        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{args.test_samples}")

    print(
        f"\nGeneration complete: "
        f"{success_count}/{args.test_samples} properties saved to "
        f"{args.property_folder}"
    )
