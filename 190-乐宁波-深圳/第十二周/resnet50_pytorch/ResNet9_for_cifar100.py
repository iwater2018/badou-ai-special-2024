import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from thop import profile
from torch.utils.data import DataLoader
from ResNet9 import resnet9

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
                                     ])

# 加载 CIFAR100 数据集
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

pretrained_model_path = 'resnet9_cifar100.pth'
model = resnet9()


def format_params(num):
    for unit in [' ', 'K', 'M', 'B']:
        if num >= 1024.0:
            num /= 1024.0
        else:
            return f"{num:.2f}{unit}"


# 遍历模型的每一层，并打印每一层的参数量
for name, param in model.named_parameters():
    if param.requires_grad:
        total_params = param.numel()
        formatted_params = format_params(total_params)
        print(f"Layer: {name} | Size: {param.size()} | Total Parameters: {formatted_params}")

total_params = sum(torch.numel(param) for param in model.parameters())
print(f"Total number of parameters: {format_params(total_params)}")

# raise 11
if os.path.exists(pretrained_model_path):
    print("Loading pretrained model...")
    model.load_state_dict(torch.load(pretrained_model_path))
else:
    pass

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 50  # 增加训练周期
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')

    # 在测试集上进行评估
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy on the test set: {100 * correct / total}%')

# 保存模型
torch.save(model.state_dict(), pretrained_model_path)
