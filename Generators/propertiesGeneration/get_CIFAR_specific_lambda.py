import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from tqdm import tqdm
import numpy as np

# ----------------------
# Import modello
# ----------------------
from training.utils.nn_models import ResNet, BasicBlock

# ----------------------
# Device
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Parametri
# ----------------------
DATASET_DIRECTORY = "./data"
CHECKPOINT_PATH = "./target_model/resnet18_fc512.pth"
NUM_POINTS = 100       # punti casuali dal test set
NUM_MC_SAMPLES = 100   # campioni Monte Carlo per punto
EPS = 0.03             # intervallo campionamento
BATCH_SIZE = 128

# ----------------------
# Dataset CIFAR-10 (solo test)
# ----------------------
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

test_set = datasets.CIFAR10(root=DATASET_DIRECTORY, train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------
# Caricamento modello ResNet18
# ----------------------
model = ResNet(
    block=BasicBlock,
    num_blocks=[2, 2, 2, 2],   # ResNet18
    num_classes=10,
    last_layer_dim=28*28,
    fc_hidden_dim=16
)

state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
model_dict = model.state_dict()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
model.load_state_dict(filtered_state_dict, strict=False)

model.to(device)
model.eval()

# ----------------------
# Calcolo min/max backbone su test set
# ----------------------
print("🔹 Propagazione test set per calcolare min/max della backbone...")
all_feats = []
with torch.no_grad():
    for images, _ in tqdm(test_loader):
        images = images.to(device)
        feats = model.forward_backbone(images)
        all_feats.append(feats.cpu())
X_test_feats = torch.cat(all_feats, dim=0)
out_min = X_test_feats.min(dim=0)[0]
out_max = X_test_feats.max(dim=0)[0]
print("✅ Min/max calcolati.")

# ----------------------
# Selezione punti casuali dal test set
# ----------------------
indices = random.sample(range(len(test_set)), NUM_POINTS)
x_samples = torch.stack([test_set[i][0] for i in indices]).to(device)

# ----------------------
# Monte Carlo e raccolta TUTTE le differenze
# ----------------------
all_diffs = []

print("🔹 Monte Carlo e calcolo differenze...")
with torch.no_grad():
    for x_s in tqdm(x_samples):
        x_s = x_s.unsqueeze(0)  # [1, 3, 32, 32]

        # Campioni Monte Carlo in [x_s-EPS, x_s+EPS]
        x_mc = x_s + (torch.rand(NUM_MC_SAMPLES, *x_s.shape[1:], device=device)*2 - 1)*EPS
        x_mc = torch.clamp(x_mc, 0.0, 1.0)

        # Propagazione SOLO backbone
        out_s = model.forward_backbone(x_s)
        out_mc = model.forward_backbone(x_mc)

        # Normalizzazione tra 0 e 1 usando min/max del test set
        out_s_norm = (out_s - out_min.to(device)) / (out_max.to(device) - out_min.to(device) + 1e-8)
        out_mc_norm = (out_mc - out_min.to(device)) / (out_max.to(device) - out_min.to(device) + 1e-8)

        # Differenze assolute
        diffs = (out_s_norm - out_mc_norm).abs().cpu().numpy().flatten()
        all_diffs.extend(diffs)

# ----------------------
# Calcolo statistiche globali
# ----------------------
all_diffs = np.array(all_diffs)
global_min = np.min(all_diffs)
global_Q1 = np.percentile(all_diffs, 25)
global_median = np.median(all_diffs)
global_Q3 = np.percentile(all_diffs, 75)
global_max = np.max(all_diffs)

print("✅ Analisi completata")
print(f"Totale punti analizzati: {all_diffs.shape[0]}")
print(f"Min: {global_min:.6f}")
print(f"Q1: {global_Q1:.6f}")
print(f"Median: {global_median:.6f}")
print(f"Q3: {global_Q3:.6f}")
print(f"Max: {global_max:.6f}")
