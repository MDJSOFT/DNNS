'''
使用现有的标记样本训练模型。
生成伪标记：使用训练好的模型对未标记样本进行预测，生成伪标记。
选择不确定性样本：从未标记样本中选择不确定性高的样本，可能基于模型的预测熵或最小置信度。
更新标记集：将高不确定性样本的伪标记添加到标记样本集中。这可以包括对每个伪标记进行验证，以确保其可靠性。
重新训练模型：使用更新后的标记样本集（包括伪标记样本）重新训练模型。
迭代过程：重复上述步骤，逐步扩展标记样本的数量。
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

# 选择伪未标记数据
num_unlabelled_samples = 6000
all_indices = set(range(len(full_dataset)))
labelled_indices_set = set(labelled_indices)
unlabelled_indices = list(all_indices - labelled_indices_set)[:num_unlabelled_samples]
unlabelled_dataset = Subset(full_dataset, unlabelled_indices)

# 2. 定义LeNet-1模型
class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.avg_pool2d(x, kernel_size=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.avg_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)  # 展平
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet1().to(device)

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
# 6. 选择不确定性样本
def select_uncertain_samples(model, dataset, n_samples=600):  # 修改为600个样本
    model.eval()
    scores = []
    with torch.no_grad():
        for images, _ in DataLoader(dataset, batch_size=64):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            uncertainty = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # 计算熵
            scores.extend(uncertainty.cpu().numpy())
    uncertain_indices = np.argsort(scores)[-n_samples:]  # 选择不确定性高的样本
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
# 主动学习循环
# 主动学习循环
for iteration in range(10):
    print(f'Iteration {iteration + 1}')

    # 训练模型
    train_model(model, labelled_loader, criterion, optimizer, epochs=5)

    # 生成伪标记
    pseudo_labels = generate_pseudo_labels(model, unlabelled_dataset)

    # 更新标记数据集
    new_indices = select_uncertain_samples(model, unlabelled_dataset, n_samples=600)  # 选择600个样本
    labelled_indices.extend(new_indices)  # 添加新样本
    labelled_indices = list(set(labelled_indices))  # 去重
    labelled_dataset = Subset(full_dataset, labelled_indices)
    labelled_loader = DataLoader(labelled_dataset, batch_size=16, shuffle=True)

    # 打印伪标记数量和扩展后的原始集数量
    num_pseudo_labels = len(pseudo_labels)
    print(f'Number of pseudo labels generated: {num_pseudo_labels}')
    print(f'Updated labelled samples: {len(labelled_indices)}')

    # 评估伪标记的正确性
    true_labels = [full_dataset[i][1] for i in unlabelled_indices]  # 获取真实标签
    correct_predictions = np.sum(np.array(pseudo_labels) == np.array(true_labels))
    accuracy = correct_predictions / num_pseudo_labels if num_pseudo_labels > 0 else 0
    print(f'Accuracy of pseudo labels: {accuracy:.2f}')


# 最后一次迭代后的模型评估
final_predictions = generate_pseudo_labels(model, full_dataset)  # 使用完整数据集生成最终预测
true_labels = [label for _, label in full_dataset]

evaluate_model(true_labels, final_predictions)


'''
主动学习和伪标记方法的区别主要体现在样本选择和使用策略上：

主动学习：

主动学习通过选择模型不确定性高的样本进行标注，优化标记数据的选择过程。标记者（或专家）只对这些样本进行标注，以提升模型的性能。
重点在于主动选择最具信息量的未标记样本，从而减少所需的标记样本数量。
伪标记：

伪标记是指使用模型对未标记数据进行预测，并将这些预测结果作为“伪标签”加入到训练集中。
重点在于利用模型的输出（即使是不完全准确的标签）来扩大训练集，提高模型的泛化能力。
总结
主动学习注重在数据标注过程中的选择，而伪标记则侧重于使用模型生成的标签来增强训练集。

'''
