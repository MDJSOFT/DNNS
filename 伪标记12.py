#使用初始模型对未标记数据进行预测，并将预测结果作为伪标签，然后将这些未标记数据与标记数据一起用于训练。
'''
始训练：用600个标记样本训练模型。
伪标记：每次迭代选择一部分未标记样本生成伪标签。
逐步扩展：在每次迭代中，将伪标记样本添加到标记样本中，并进行训练。
'''

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

# 加载MNIST数据集
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 随机选择带标记数据
num_labelled_samples = 600
labelled_indices = random.sample(range(len(full_dataset)), num_labelled_samples)
labelled_dataset = Subset(full_dataset, labelled_indices)
labelled_loader = DataLoader(labelled_dataset, batch_size=16, shuffle=True)

# 选择未标记数据
num_unlabelled_samples = 6000
all_indices = set(range(len(full_dataset)))
labelled_indices_set = set(labelled_indices)
unlabelled_indices = list(all_indices - labelled_indices_set)
unlabelled_dataset = Subset(full_dataset, unlabelled_indices)

class LeNet4(nn.Module):
    def __init__(self):
        super(LeNet4, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.act3 = nn.Tanh()

        self.fc1 = nn.Linear(120, 84)
        self.act4 = nn.Tanh()
        self.fc2 = nn.Linear(84, 10)  # 10个类别

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.act3(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.act4(self.fc1(x))
        x = self.fc2(x)
        return x

# 替换加载的模型部分
model = LeNet4()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


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

## 在每次迭代后评估伪标记的正确性
for iteration in range(10):
    print(f'Iteration {iteration + 1}')

    # 训练模型
    train_model(model, labelled_loader, criterion, optimizer, epochs=5)

    # 生成伪标记
    pseudo_labels = generate_pseudo_labels(model, unlabelled_dataset)

    # 每次迭代选择600个伪标记
    selected_indices = random.sample(range(len(unlabelled_indices)), 600)
    labelled_indices.extend([unlabelled_indices[i] for i in selected_indices])  # 添加伪标记样本的索引

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



















'''
主动学习部分：
select_combined_uncertain_samples 函数用于选择不确定性高的样本，这体现了主动学习的策略，主动选择最具信息量的未标记样本。
在每次迭代中，主动学习会从未标记数据集中选择样本，并要求这些样本进行标注（或使用真实标签），然后将其加入到标记数据中。
伪标记部分：

代码的伪标记部分并没有明确展示，但可以推测通过模型预测未标记样本并用其预测结果作为标签进行训练。伪标记通常是在模型训练后，对未标记样本进行推理并将其结果作为“伪标签”来扩展训练集。

'''
