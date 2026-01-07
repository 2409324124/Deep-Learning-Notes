# 首先导入必要的库
import matplotlib.pyplot as plt  # 添加这一行

# 然后设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 优先用 Windows 自带中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 数据加载（完整，包括训练集和测试集）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("加载训练集和测试集...")
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=False,  # 如果文件已手动放好，用 False；否则 True
    transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=False, 
    transform=transform
)

# 训练集加载器（关键！用于训练循环）
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 测试集加载器（用于评估）
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print("数据集加载完成！")


# 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = SimpleCNN().to(device)

# 训练（快速 5 epoch）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("开始训练...")
train_losses = []
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:  # 注意：需要 train_loader，如果没有请先定义
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"第 {epoch+1} 轮完成，平均损失: {avg_loss:.4f}")

print("训练完成！")

# 测试 + 收集所有预测（关键，用于混淆矩阵）
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"测试准确率: {accuracy:.2f}%")

# 漂亮混淆矩阵（seaborn）
print("\n绘制混淆矩阵...")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('混淆矩阵（行:真实标签，列:预测标签）')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.show()

print("可视化完成！")