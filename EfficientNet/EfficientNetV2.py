import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_v2_s
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from torchvision import transforms


# =================== 超参数 ===================
data_dir = "./JUST"
batch_size = 32
num_epochs = 100
learning_rate = 0.00001
weight_decay = 1e-5
train_split = 0.8
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================== 数据集加载 ===================
# 数据目录
data_dir = "./JUST"
real_dir = os.path.join(data_dir, "Real")
fake_dir = os.path.join(data_dir, "Fake")

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 使用 ImageFolder 加载数据
dataset = ImageFolder(root=data_dir, transform=data_transforms)

# 划分训练集和测试集
data_size = len(dataset)
train_size = int(data_size * train_split)
test_size = data_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# =================== 模型定义 ===================
class EfficientNetV2(nn.Module):
    def __init__(self):
        super(EfficientNetV2, self).__init__()
        self.model = efficientnet_v2_s(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)  # 二分类任务

    def forward(self, x):
        return self.model(x)

model = EfficientNetV2().to(device)


# =================== 损失函数和优化器 ===================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# =================== 训练和测试函数 ===================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# =================== 训练过程 ===================
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}")


# =================== 保存模型 ===================
model_path = "Efficientnetv2.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


# =================== 绘制并保存结果 ===================
plt.figure(figsize=(10, 10))

# 绘制 Loss 曲线
plt.subplot(2, 1, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(test_losses, label='Test Loss', color='orange')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制 Accuracy 曲线
plt.subplot(2, 1, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(test_accuracies, label='Test Accuracy', color='orange')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("metrics_curve.png")