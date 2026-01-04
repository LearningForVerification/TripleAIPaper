import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import pandas as pd

from training.utils.nn_models import ResNet, BasicBlock

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIRECTORY = "./data"
os.makedirs(DATASET_DIRECTORY, exist_ok=True)

# --------------------------------------------------
# Build model (ResNet18)
# --------------------------------------------------
model = ResNet(
    block=BasicBlock,
    num_blocks=[2, 2, 2, 2],   # ResNet18
    num_classes=10,            # CIFAR-10
    last_layer_dim=28*28,       # come da checkpoint
    fc_hidden_dim=16
)

# --------------------------------------------------
# Load state_dict (solo chiavi compatibili)
# --------------------------------------------------
state_dict = torch.load(r"target_model/resnet18_fc512.pth", map_location=device)
model_dict = model.state_dict()
filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if k in model_dict and model_dict[k].shape == v.shape
}
model.load_state_dict(filtered_state_dict, strict=False)
model.to(device)
model.eval()

# --------------------------------------------------
# Transforms
# --------------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# --------------------------------------------------
# CIFAR-10 train & test
# --------------------------------------------------
train_dataset = datasets.CIFAR10(root=DATASET_DIRECTORY, train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root=DATASET_DIRECTORY, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# --------------------------------------------------
# Feature extraction
# --------------------------------------------------
def extract_features(loader):
    feats = []
    labels = []
    with torch.no_grad():
        for images, y in tqdm(loader):
            images = images.to(device)
            f = model.forward_backbone(images)
            feats.append(f.cpu())
            labels.append(y)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

X_train, y_train = extract_features(train_loader)
X_test, y_test = extract_features(test_loader)

print("Train features:", X_train.shape)
print("Test features:", X_test.shape)

# --------------------------------------------------
# Salvataggio diretto in CSV
# --------------------------------------------------
def save_to_csv(X, y, csv_file):
    df = pd.DataFrame(torch.cat([X, y.unsqueeze(1)], dim=1).numpy())
    df.to_csv(csv_file, index=False)

train_csv = os.path.join(DATASET_DIRECTORY, "custom_train.csv")
test_csv = os.path.join(DATASET_DIRECTORY, "custom_test.csv")

save_to_csv(X_train, y_train, train_csv)
save_to_csv(X_test, y_test, test_csv)

print("CSV creati correttamente:")
print("Train:", train_csv)
print("Test:", test_csv)
