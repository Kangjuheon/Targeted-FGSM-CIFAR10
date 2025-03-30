import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FGSM Targeted Attack
def fgsm_targeted_attack(model, x, target, eps):
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True
    output = model(x_adv)
    loss = F.cross_entropy(output, target.to(device))  # 목표 클래스를 향한 손실
    model.zero_grad()
    loss.backward()
    grad_sign = x_adv.grad.data.sign()
    x_adv = x_adv - eps * grad_sign  # 👈 부호 반대
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

# 목표 클래스 지정
def generate_random_targets(labels, num_classes=10):
    return torch.tensor([(label + random.randint(1, 9)) % num_classes for label in labels])

# 평가
def evaluate_targeted_attack(model, loader, eps):
    model.eval()
    success = 0
    total = 0
    for x, label in tqdm(loader, desc="🎯 FGSM Targeted Evaluation"):
        x, label = x.to(device), label.to(device)
        target = generate_random_targets(label).to(device)
        x_adv = fgsm_targeted_attack(model, x, target, eps)
        with torch.no_grad():
            output = model(x_adv)
            pred = output.argmax(dim=1)
            success += pred.eq(target).sum().item()
            total += len(x)
    return 100 * success / total

# 메인
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 10)
    model = model.to(device)

    eps = 0.03
    acc = evaluate_targeted_attack(model, test_loader, eps)
    print(f"\n[FGSM Targeted Attack Success Rate] eps={eps} → {acc:.2f}%")
