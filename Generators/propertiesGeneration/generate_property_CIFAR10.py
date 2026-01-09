import os
import csv
import numpy as np
import torch

# ============================================================
# Dataset CUSTOM
# ============================================================
class CustomFeatureDataset:
    def __init__(self, csv_file, normalize=False):
        df = csv.reader(open(csv_file))
        header = next(df)  # salta header

        X, y = [], []
        for row in df:
            X.append([float(x) for x in row[:-1]])  # features
            y.append(int(float(row[-1])))           # label

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        if normalize:
            # Normalizzazione feature-wise in [0,1]
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)
            denom = X_max - X_min
            denom[denom == 0] = 1.0
            X = (X - X_min) / denom
            print("Normalized!")

        self.X = X
        self.y = y

        # Stampa dimensioni e range globale
        print(f"Dataset shape: {self.X.shape}")
        print(f"Labels shape: {self.y.shape}")
        print(f"Global min of X: {self.X.min()}")
        print(f"Global max of X: {self.X.max()}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# Generazione proprietà VNNLIB (robustezza locale)
# ============================================================
def generate_local_robustness_property(
    input_sample,
    noise_level,
    correct_label,
    property_path,
    num_classes=10,
    total_properties=None,
    sample_idx=None
):
    """
    Genera un file .vnnlib per la robustezza locale.
    Ogni assert di output è separato per ABCROWN.
    """
    os.makedirs(os.path.dirname(property_path), exist_ok=True)
    flat_input = input_sample.flatten()

    with open(property_path, "w") as f:
        # ----------------------------------------------------
        # Commenti informativi
        # ----------------------------------------------------
        if total_properties is not None:
            f.write(f"; Total properties: {total_properties}\n")
        if sample_idx is not None:
            f.write(f"; Property index: {sample_idx}\n")
        f.write(f"; Correct label: {correct_label}\n")
        f.write(f"; Epsilon (L_inf): {noise_level}\n\n")

        # ----------------------------------------------------
        # Variabili input
        # ----------------------------------------------------
        for i in range(flat_input.size):
            f.write(f"(declare-const X_{i} Real)\n")

        # ----------------------------------------------------
        # Variabili output
        # ----------------------------------------------------
        for i in range(num_classes):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write("\n")

        # ----------------------------------------------------
        # Vincoli sugli input (L_inf ball)
        # ----------------------------------------------------
        for i, val in enumerate(flat_input):
            if val < 0.0 or val > 1.0:
                raise ValueError(f"Input value out of range [0,1] at index {i}: {val}")

            lower = max(0.0, val - noise_level)
            upper = min(1.0, val + noise_level)

            if lower > upper:
                raise ValueError(f"Incoherent bounds at index {i}: lower={lower} > upper={upper}")

            f.write(f"(assert (>= X_{i} {lower:.10f}))\n")
            f.write(f"(assert (<= X_{i} {upper:.10f}))\n")

        f.write("\n")

        # ----------------------------------------------------
        # Negazione della robustezza locale (output constraints)
        # ∃ i ≠ y : Y_i ≥ Y_y
        # ----------------------------------------------------
        for i in range(num_classes):
            if i != correct_label:
                f.write(f"(assert (>= Y_{i} Y_{correct_label}))\n")


# ============================================================
# Script principale
# ============================================================
if __name__ == "__main__":
    CSV_PATH = "./custom_test.csv"                          # percorso CSV
    PROPERTY_FOLDER = "./properties/CIFAR_CUSTOM/0.03"      # cartella output
    EPSILON = 0.03                                           # perturbazione L_inf
    NUM_CLASSES = 10                                         # numero classi
    NORMALIZE = True                                         # normalizzazione

    dataset = CustomFeatureDataset(CSV_PATH, normalize=NORMALIZE)
    os.makedirs(PROPERTY_FOLDER, exist_ok=True)

    total_properties = 100

    for idx in range(total_properties):
        input_np, label = dataset[idx]
        prop_path = os.path.join(
            PROPERTY_FOLDER,
            f"sample_{idx:04d}_label_{label}_eps_{EPSILON:.4f}.vnnlib"
        )
        generate_local_robustness_property(
            input_np,
            EPSILON,
            label,
            prop_path,
            num_classes=NUM_CLASSES,
            total_properties=total_properties,
            sample_idx=idx
        )

    print(f"✅ Generated {total_properties} properties in {PROPERTY_FOLDER}")
