import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. 数据准备与增强
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Fashion MNIST 是灰度图像
])

# 加载Fashion MNIST数据集
full_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# 随机选择带标记数据
num_labelled_samples = 1000
labelled_indices = random.sample(range(len(full_dataset)), num_labelled_samples)
labelled_dataset = Subset(full_dataset, labelled_indices)
labelled_loader = DataLoader(labelled_dataset, batch_size=16, shuffle=True)

# 选择伪未标记数据
num_unlabelled_samples = 10000
all_indices = set(range(len(full_dataset)))
labelled_indices_set = set(labelled_indices)
unlabelled_indices = list(all_indices - labelled_indices_set)[:num_unlabelled_samples]
unlabelled_dataset = Subset(full_dataset, unlabelled_indices)

# 2. 定义ResNet-15模型
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16  # ResNet-15 使用较小的输入通道数
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)  # 输入通道改为1
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

# 定义ResNet-15
def ResNet15():
    return ResNet(BasicBlock, [3, 3, 3], num_classes=10)

# 替换原有的模型定义为 ResNet-15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet15().to(device)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 模型训练
def train_model(model, data_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(data_loader):.4f}')

# 5. 伪标记生成
def generate_pseudo_labels(model, dataset):
    model.eval()
    pseudo_labels = []
    with torch.no_grad():
        for images, _ in DataLoader(dataset, batch_size=64):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            pseudo_labels.extend(predicted.cpu().numpy())
    return pseudo_labels

# 6. 选择不确定性样本
def select_uncertain_samples(model, dataset, n_samples=600):
    model.eval()
    scores = []
    with torch.no_grad():
        for images, _ in DataLoader(dataset, batch_size=64):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            uncertainty = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            scores.extend(uncertainty.cpu().numpy())
    uncertain_indices = np.argsort(scores)[-n_samples:]
    return uncertain_indices

# 定义评估函数
def evaluate_model(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

# 主动学习循环
for iteration in range(10):
    print(f'Iteration {iteration + 1}')

    # 训练模型
    train_model(model, labelled_loader, criterion, optimizer, epochs=5)

    # 生成伪标记
    pseudo_labels = generate_pseudo_labels(model, unlabelled_dataset)

    # 更新标记数据集
    new_indices = select_uncertain_samples(model, unlabelled_dataset, n_samples=1000)
    labelled_indices.extend(new_indices)
    labelled_indices = list(set(labelled_indices))  # 去重
    labelled_dataset = Subset(full_dataset, labelled_indices)
    labelled_loader = DataLoader(labelled_dataset, batch_size=16, shuffle=True)

    # 打印伪标记数量和扩展后的原始集数量
    num_pseudo_labels = len(pseudo_labels)
    print(f'Number of pseudo labels generated: {num_pseudo_labels}')
    print(f'Updated labelled samples: {len(labelled_indices)}')

    # 评估伪标记的正确性
    true_labels = [full_dataset[i][1] for i in unlabelled_indices]
    correct_predictions = np.sum(np.array(pseudo_labels) == np.array(true_labels))
    accuracy = correct_predictions / num_pseudo_labels if num_pseudo_labels > 0 else 0
    print(f'Accuracy of pseudo labels: {accuracy:.2f}')

# 最后一次迭代后的模型评估
final_predictions = generate_pseudo_labels(model, full_dataset)
true_labels = [label for _, label in full_dataset]

evaluate_model(true_labels, final_predictions)
