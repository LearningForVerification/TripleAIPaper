import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import os
import time  # <-- Aggiungi import per time

# ------------------------------------------------------------
# 1. Configurazione
# ------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 1
LR = 0.1

# Dimensioni del layer4 da testare (partendo da 30 e salendo)
LAST_LAYER_DIMS = [5, 10, 30, 64, 128, 256, 512, 1024, 2048]
LAST_LAYER_DIMS = [1024]

RESULTS_FILE = 'resnet_variable_dim_results_incremental.csv'
MODELS_FOLDER = 'models'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device in uso: {DEVICE}")

# Crea la cartella models se non esiste
os.makedirs(MODELS_FOLDER, exist_ok=True)
print(f"Cartella modelli: {MODELS_FOLDER}")

# Se il file esiste già, lo rimuoviamo per partire puliti (opzionale, commenta se vuoi appendere a vecchi run)
if os.path.exists(RESULTS_FILE):
    os.remove(RESULTS_FILE)
    print(f"File {RESULTS_FILE} esistente rimosso. Inizio nuovo log.")

# ------------------------------------------------------------
# 2. Dataset e Trasformazioni
# ------------------------------------------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


# ------------------------------------------------------------
# 3. ResNet Modificata (Parametrica su ultimo layer)
# ------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, last_layer_dim=512):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        # QUI la modifica: layer4 usa last_layer_dim invece di 512 fisso
        self.layer4 = self._make_layer(block, last_layer_dim, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Il FC layer si adatta alla dimensione di output di layer4
        self.fc = nn.Linear(last_layer_dim * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet18(last_layer_dim=512):
    return ResNet(BasicBlock, [2, 2, 2, 2], last_layer_dim=last_layer_dim)


# ------------------------------------------------------------
# 4. Training Loop
# ------------------------------------------------------------
def train_epoch(model, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return running_loss / len(trainloader), 100. * correct / total


def evaluate(model, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return running_loss / len(testloader), 100. * correct / total


# ------------------------------------------------------------
# 5. Main Execution
# ------------------------------------------------------------
if __name__ == "__main__":

    print(f"Inizio esperimenti sulle dimensioni: {LAST_LAYER_DIMS}")

    for i, dim in enumerate(LAST_LAYER_DIMS):
        print(f"\n--- Training Model {i + 1}/{len(LAST_LAYER_DIMS)}: Last Layer Dim = {dim} ---")

        model = ResNet18(last_layer_dim=dim).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

        # Inizia il timing per questo modello
        model_start_time = time.time()  # <-- Inizio tempo per l'intero modello

        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()  # <-- Inizio tempo per questa epoca

            train_loss, train_acc = train_epoch(model, optimizer, criterion)
            test_loss, test_acc = evaluate(model, criterion)
            scheduler.step()

            epoch_time = time.time() - epoch_start_time  # <-- Calcola tempo epoca

            if epoch % 5 == 0 or epoch == EPOCHS:
                print(
                    f"Ep {epoch}/{EPOCHS} | Tr Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Te Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | Time: {epoch_time:.2f}s")

            # --- SALVATAGGIO INCREMENTALE ---
            current_stats = {
                'last_layer_dim': [dim],
                'epoch': [epoch],
                'train_loss': [train_loss],
                'train_acc': [train_acc],
                'test_loss': [test_loss],
                'test_acc': [test_acc],
                'epoch_time_seconds': [epoch_time]  # <-- Aggiungi tempo epoca
            }

            df_epoch = pd.DataFrame(current_stats)

            # Se il file non esiste, scriviamo l'header. Se esiste, appendiamo senza header.
            header_flag = not os.path.exists(RESULTS_FILE)
            df_epoch.to_csv(RESULTS_FILE, mode='a', header=header_flag, index=False)
            # ---------------------------------

        total_model_time = time.time() - model_start_time  # <-- Tempo totale per il modello
        print(f"Tempo totale di training per dim {dim}: {total_model_time:.2f}s")

        # --- SALVATAGGIO MODELLI (PyTorch e ONNX) ---
        model_name = f"resnet18_dim{dim}"

        # Salvataggio in formato PyTorch (.pth)
        torch_path = os.path.join(MODELS_FOLDER, f"{model_name}.pth")
        torch.save(model.state_dict(), torch_path)
        print(f"Modello salvato (PyTorch): {torch_path}")

        # Salvataggio in formato ONNX
        onnx_path = os.path.join(MODELS_FOLDER, f"{model_name}.onnx")
        dummy_input = torch.randn(1, 3, 32, 32).to(DEVICE)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Modello salvato (ONNX): {onnx_path}")
        # --------------------------------------------

        del model
        del optimizer
        torch.cuda.empty_cache()

    print(f"\nTraining completato. Tutti i dati sono stati salvati in: {RESULTS_FILE}")
    print(f"Modelli salvati in: {MODELS_FOLDER}")
