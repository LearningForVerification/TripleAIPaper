import os
import torch
from torch import nn, Tensor
from torchvision import datasets, transforms

from training.utils.rs_loss_regularizer import calculate_rs_loss_regularizer_conv
import onnx
import onnx2pytorch
from training.utils.dataset import get_data_loader
from auto_LiRPA import BoundedModule




def count_unstable_nodes(model_path: str, dummy_input, architecture_tuple, input_batch: Tensor, eps: float = 0.03, device="cpu"):
    """
    Carica un modello ONNX, lo converte in PyTorch e calcola i neuroni instabili.
    """
    # Carica il modello onnx
    onnx_model = onnx.load(model_path)

    # Converte il modello in pytorch (con onnx2pytorch)
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model)


    # Wrappa in modello LIRPA
    lirpa_model = BoundedModule(
        pytorch_model,
        dummy_input,   # serve un input fittizio per tracciare la rete
        device=device
    )

    # Calcola rs_loss e numero neuroni instabili
    rs_loss, n_unstable_nodes = calculate_rs_loss_regularizer_conv(
        lirpa_model, architecture_tuple, (input_batch, 0), eps, method="ibp", normalized=True
    )

    return n_unstable_nodes.item() if torch.is_tensor(n_unstable_nodes) else n_unstable_nodes


# 👇 qui imposti il path della cartella con i modelli
MODELS_DIR = "over_param"
EPS = 0.03

def main():
    # Prende tutti i file .onnx nella cartella
    model_files = [
        os.path.join(MODELS_DIR, f)
        for f in os.listdir(MODELS_DIR)
        if f.endswith(".onnx")
    ]

    if not model_files:
        print(f"[ERROR] Nessun file .onnx trovato in {MODELS_DIR}")
        return

    # Carica il test set MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten (28x28 → 784)
    ])

    train_loader, test_loader, dummy_input, input_dim, output_dim = get_data_loader(
        "MNIST", 128, 128, input_flattened=False)

    unstable_counts = []
    for model_path in model_files:

        model_unstable_vals = []

        for inputs, _ in test_loader:
            unstable = count_unstable_nodes(model_path, dummy_input, 7, inputs, EPS)
            model_unstable_vals.append(unstable)

        avg_unstable_model = sum(model_unstable_vals) / len(model_unstable_vals)
        print(f"Modello: {model_path} → media neuroni instabili: {avg_unstable_model:.2f}")
        unstable_counts.append(avg_unstable_model)


    if unstable_counts:
        avg_unstable_all = sum(unstable_counts) / len(unstable_counts)
        print(f"\nMedia neuroni instabili (su tutti i modelli): {avg_unstable_all:.2f}")
    else:
        print("Nessun modello valido processato.")


if __name__ == "__main__":
    main()
