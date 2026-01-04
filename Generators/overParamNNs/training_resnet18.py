import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import os
import time


# ------------------------------------------------------------
# 1. Configurazione
# ------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 300
LR = 0.01

# Lista dei layer intermedi da testare
FC_HIDDEN_DIMS = [8, 16, 32, 64, 128, 256, 512]
FC_HIDDEN_DIMS = [512]

LAST_LAYER_DIM = 28*28

RESULTS_FILE = 'resnet_results.csv'
MODELS_FOLDER = 'models'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODELS_FOLDER, exist_ok=True)
if os.path.exists(RESULTS_FILE):
    os.remove(RESULTS_FILE)

print(f"Device: {DEVICE}")
print(f"Cartella modelli: {MODELS_FOLDER}")

# ------------------------------------------------------------
# 2. Dataset e trasformazioni
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
# 3. ResNet con 2 layer FC
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
    def __init__(self, block, num_blocks, num_classes=10, last_layer_dim=LAST_LAYER_DIM, fc_hidden_dim=256):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.last_layer_dim = last_layer_dim
        self.fc_hidden_dim = fc_hidden_dim

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, last_layer_dim, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Due layer FC con ReLU intermedia
        self.fc1 = nn.Linear(last_layer_dim * block.expansion, fc_hidden_dim)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes)

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
        out = self.fc1(out)
        out = self.relu_fc(out)
        out = self.fc2(out)
        return out

def ResNet18(last_layer_dim=32, fc_hidden_dim=256):
    return ResNet(BasicBlock, [2,2,2,2], last_layer_dim=last_layer_dim, fc_hidden_dim=fc_hidden_dim)

# ------------------------------------------------------------
# 4. Train/Val functions
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
    return running_loss / len(trainloader), 100.*correct/total

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
    return running_loss / len(testloader), 100.*correct/total

# ------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------
def main(init_model_path=None):
    for fc_hidden in FC_HIDDEN_DIMS:
        print(f"\n--- Training con fc_hidden_dim = {fc_hidden} ---")
        model = ResNet18(last_layer_dim=LAST_LAYER_DIM, fc_hidden_dim=fc_hidden).to(DEVICE)

        # Carica backbone se fornito
        if init_model_path:
            print(f"Carico pesi backbone da: {init_model_path}")
            backbone_state = torch.load(init_model_path)
            model_state = model.state_dict()
            for k in backbone_state:
                if "fc1" not in k and "fc2" not in k and k in model_state:
                    model_state[k] = backbone_state[k]
            model.load_state_dict(model_state)
            for name, param in model.named_parameters():
                if "fc1" not in name and "fc2" not in name:
                    param.requires_grad = False
            print("Backbone caricato e congelato. FC1 e FC2 addestrabili.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

        epoch_times = []
        for epoch in range(1, EPOCHS+1):
            start_time = time.time()
            train_loss, train_acc = train_epoch(model, optimizer, criterion)
            test_loss, test_acc = evaluate(model, criterion)
            scheduler.step()
            epoch_times.append(time.time() - start_time)

        mean_epoch_time = sum(epoch_times)/len(epoch_times)

        # Salvataggio statistiche
        stats = {
            'fc_hidden_dim': fc_hidden,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'mean_epoch_time_seconds': mean_epoch_time
        }
        df = pd.DataFrame([stats])
        df.to_csv(RESULTS_FILE, mode='a', header=not os.path.exists(RESULTS_FILE), index=False)

        # Salvataggio modelli
        model_name = f"resnet18_fc{fc_hidden}"
        torch.save(model.state_dict(), os.path.join(MODELS_FOLDER, f"{model_name}.pth"))
        dummy_input = torch.randn(1,3,32,32).to(DEVICE)
        torch.onnx.export(model, dummy_input, os.path.join(MODELS_FOLDER, f"{model_name}.onnx"),
                          export_params=True, opset_version=11, do_constant_folding=True,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0:'batch_size'}, 'output': {0:'batch_size'}})

        del model
        torch.cuda.empty_cache()

    print(f"\nTraining completato. Risultati in {RESULTS_FILE}. Modelli in {MODELS_FOLDER}")

if __name__ == "__main__":
    INIT_MODEL_PATH = None
    main(init_model_path=INIT_MODEL_PATH)
