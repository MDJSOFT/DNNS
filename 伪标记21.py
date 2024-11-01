
import torch.nn.functional as F
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
    transforms.Resize((32, 32)),  # LeNet-1 输入大小为 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载SVHN数据集
full_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)

# 随机选择带标记数据
num_labelled_samples = 1000
labelled_indices = random.sample(range(len(full_dataset)), num_labelled_samples)
labelled_dataset = Subset(full_dataset, labelled_indices)
labelled_loader = DataLoader(labelled_dataset, batch_size=16, shuffle=True)

# 选择未标记数据
num_unlabelled_samples = 10000
all_indices = set(range(len(full_dataset)))
labelled_indices_set = set(labelled_indices)
unlabelled_indices = list(all_indices - labelled_indices_set)
unlabelled_dataset = Subset(full_dataset, unlabelled_indices)

# 2. 定义Simplified VGG16模型
class SimplifiedVGG16(nn.Module):
    def __init__(self, num_classes=10, image_size=(28, 28, 1)):
        super(SimplifiedVGG16, self).__init__()
        # 由于图像是单通道的，我们使用第一个卷积层来转换通道数
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # 8192

        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)  # 将特征展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# 替换原有的模型定义为 SimplifiedVGG16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimplifiedVGG16().to(device)

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

    # 每次迭代选择600个伪标记
    selected_indices = random.sample(range(len(unlabelled_indices)), 1000)
    selected_unlabelled_indices = [unlabelled_indices[i] for i in selected_indices]
    labelled_indices.extend(selected_unlabelled_indices)  # 添加伪标记样本的索引

    # 更新标记数据集
    labelled_dataset = Subset(full_dataset, labelled_indices)
    labelled_loader = DataLoader(labelled_dataset, batch_size=16, shuffle=True)

    # 打印伪标记数量和扩展后的原始集数量
    num_pseudo_labels = len(selected_indices)
    print(f'Number of pseudo labels generated: {num_pseudo_labels}')
    print(f'Updated labelled samples: {len(labelled_indices)}')

    # 评估伪标记的正确性
    true_labels = [full_dataset[i][1] for i in unlabelled_indices]  # 获取真实标签
    selected_true_labels = [true_labels[i] for i in selected_indices]  # 获取选中样本的真实标签
    correct_predictions = np.sum(np.array(pseudo_labels)[selected_indices] == np.array(selected_true_labels))
    accuracy = correct_predictions / num_pseudo_labels if num_pseudo_labels > 0 else 0
    print(f'Accuracy of pseudo labels: {accuracy:.2f}')

# 最后一次迭代后的模型评估
final_predictions = generate_pseudo_labels(model, full_dataset)
true_labels = [label for _, label in full_dataset]

evaluate_model(true_labels, final_predictions)

