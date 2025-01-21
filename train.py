import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import LightweightCNN
from torch.utils.data import DataLoader
import tqdm
from torch.amp import GradScaler, autocast
import numpy as np
import torch.nn.functional as F

device = "cuda"

num_epochs = 400
batch_size = 256
learning_rate = 0.002
beta = 0.4

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),  # Add random erasing
])

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        ),
    ]
)

trainset = torchvision.datasets.CIFAR100(
    root="./cifar-100-python", train=True, download=True, transform=transform_train
)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR100(
    root="./cifar-100-python", train=False, download=True, transform=transform_test
)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

model = LightweightCNN().to(device)
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    total_steps=num_epochs * len(trainloader),
    pct_start=0.3,
    anneal_strategy='cos'
)
scaler = GradScaler('cuda')

best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(tqdm.tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        inputs, labels = inputs.to(device), labels.to(device)

        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(inputs.size(0)).to(device)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[rand_index]
        labels_a, labels_b = labels, labels[rand_index]

        optimizer.zero_grad()
        
        with autocast('cuda'):
            outputs = model(mixed_inputs)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {train_loss/(i+1):.3f} Train Acc: {100.*correct/total:.3f}")

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm.tqdm(testloader, desc="Test")):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    if (correct/total) > best_acc:
        torch.save(model.state_dict(), "best_model.pth")
        best_acc = correct/total
        print(f"Best Model Saved. Acc: {100.*correct/total:.3f}")

    print(f"Epoch {epoch+1}/{num_epochs} Test Loss: {test_loss/(i+1):.3f} Test Acc: {100.*correct/total:.3f}")