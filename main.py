import torch
import torchvision
import time
import onnx
import onnxruntime
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
val_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

model_path = "efficientnet_b3_rwightman-b3899882.pth"

model = torchvision.models.efficientnet_b3(weights=None)  # Без предупреждений

state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)

model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


def accuracy(net, loader):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (net(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


print(f"Всего параметров: {sum(p.numel() for p in model.parameters()):,}")
print("Начинаем обучение...")

if __name__ == '__main__':
    for epoch in range(3):
        model.train()
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}") as pbar:
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())

        val_acc = accuracy(model, val_loader)
        print(f"Epoch {epoch + 1}  val-acc={val_acc:.3f}")