import time
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import cv2

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


#2. 熵选择。选择样本时，计算模型对每个样本的预测熵。熵越高，表示模型越不确定，因此可以选择熵值最高的样本。

# 1. 数据准备与增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjust for 3 channels
])


# 加载CIFAR-10数据集
full_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# 随机选择带标记数据，减少数量
num_labelled_samples = 600  # 带标记数据数量
labelled_indices = random.sample(range(len(full_dataset)), num_labelled_samples)  # 随机选择带标记样本索引
labelled_dataset = Subset(full_dataset, labelled_indices)  # 创建带标记数据子集
labelled_loader = DataLoader(labelled_dataset, batch_size=16, shuffle=True)

# 选择伪未标记数据
num_unlabelled_samples = 6000  # 伪未标记数据数量
all_indices = set(range(len(full_dataset)))  # 所有样本的索引
labelled_indices_set = set(labelled_indices)  # 带标记样本的索引

# 从训练集中选择伪未标记数据
unlabelled_indices = list(all_indices - labelled_indices_set)[:num_unlabelled_samples]
unlabelled_dataset = Subset(full_dataset, unlabelled_indices)  # 创建伪未标记数据子集


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet15(nn.Module):
    def __init__(self):
        super(ResNet15, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 16, 5)  # 5 个 BasicBlock
        self.layer2 = self._make_layer(BasicBlock, 32, 5, stride=2)  # 5 个 BasicBlock
        self.layer3 = self._make_layer(BasicBlock, 64, 5, stride=2)  # 5 个 BasicBlock
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)  # 假设有10个类

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 创建模型实例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet15().to(device)


# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()# 定义交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001) # 使用Adam优化器

# 4. 模型训练
for epoch in range(10):# 训练10个epoch
    model.train()# 设置模型为训练模式
    for images, labels in labelled_loader:# 遍历带标记的数据加载器
        images, labels = images.to(device), labels.to(device)# 将数据移动到计算设备
        optimizer.zero_grad()# 清零梯度
        outputs = model(images) # 前向传播
        loss = criterion(outputs, labels) # 计算损失
        loss.backward()# 反向传播
        optimizer.step()# 更新模型参数

# 5. 蜕变体生成
def create_mutant(original_model):
    mutated_model = copy.deepcopy(original_model)

    for name, layer in list(mutated_model.named_children()):  # Use list to avoid mutation during iteration
        if isinstance(layer, nn.Conv2d):
            layer.kernel_size = random.choice([(3, 3), (5, 5)])
            layer.out_channels = random.choice(range(int(layer.out_channels * 0.75), int(layer.out_channels * 1.25)))
            # Create new layer if needed instead of modifying in place

    return mutated_model




# 生成变异体
mutants = [create_mutant(model) for _ in range(3)]
test_loader = DataLoader(datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform), batch_size=64)

# 6. 变异体评估
def evaluate_mutant(mutant):
    mutant.eval()
    all_predictions = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.time()  # 记录推理开始时间
            outputs = mutant(images)
            inference_time = time.time() - start_time  # 计算推理时间

            inference_times.append(inference_time)
            _, predicted = torch.max(outputs.data, 1)

            # 保存预测和真实标签
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算各项指标
    accuracy = accuracy_score(all_labels, all_predictions) * 100  # 准确率
    precision = precision_score(all_labels, all_predictions, average='weighted')  # 精确率
    recall = recall_score(all_labels, all_predictions, average='weighted')  # 召回率
    f1 = f1_score(all_labels, all_predictions, average='weighted')  # F1 分数
    conf_matrix = confusion_matrix(all_labels, all_predictions)  # 混淆矩阵
    avg_inference_time = sum(inference_times) / len(inference_times)  # 平均推理时间

    return accuracy, precision, recall, f1, conf_matrix, avg_inference_time


# 7. 主动学习：选择不确定性高的样本
# 确保在选择不确定性样本时使用 no_grad
def select_combined_uncertain_samples(model, dataset, n_samples=20):
    model.eval()
    scores = []
    with torch.no_grad():
        for images, _ in DataLoader(dataset, batch_size=64):
            outputs = model(images.to(device))
            probs = F.softmax(outputs, dim=1)
            least_confident = torch.max(probs, dim=1)[0]
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            score = least_confident * entropy
            scores.extend(score.cpu().numpy())
    uncertain_indices = np.argsort(scores)[:n_samples]
    return uncertain_indices




# 8. 特征推导与标签推断
# 特征提取
# 定义类别标签

# 特征提取：计算颜色直方图
def extract_color_features(image):
    image_np = image.numpy().transpose(1, 2, 0)  # 转换为 HWC 格式
    hist = cv2.calcHist([image_np], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # 归一化直方图
    return hist.astype(np.float32)  # 确保是 NumPy 数组


# 标签推断（示例规则）
def infer_label_based_on_rules(color_features, true_label=None):
    mean_color = np.mean(color_features)

    # 根据颜色特征的均值进行简单的类别推断
    if mean_color < 0.1:
        inferred_label = 0  # 假设为"airplane"
    elif mean_color < 0.2:
        inferred_label = 1  # 假设为"automobile"
    elif mean_color < 0.3:
        inferred_label = 2  # 假设为"bird"
    elif mean_color < 0.4:
        inferred_label = 3  # 假设为"cat"
    elif mean_color < 0.5:
        inferred_label = 4  # 假设为"deer"
    elif mean_color < 0.6:
        inferred_label = 5  # 假设为"dog"
    elif mean_color < 0.7:
        inferred_label = 6  # 假设为"frog"
    elif mean_color < 0.8:
        inferred_label = 7  # 假设为"horse"
    elif mean_color < 0.9:
        inferred_label = 8  # 假设为"ship"
    else:
        inferred_label = 9  # 假设为"truck"

        # 假设没有匹配的特征，返回真实标签
    return  true_label


# 主动学习循环
# 主动学习循环
# 主动学习循环
for iteration in range(10):
    new_indices = select_combined_uncertain_samples(model, unlabelled_dataset, n_samples=600)

    # 直接使用真实标签替换推断标签
    inferred_labels = [full_dataset[idx][1] for idx in new_indices]  # 使用真实标签

    # 输出每个选择样本及其真实标签
    for idx in new_indices:
        color_features = extract_color_features(full_dataset[idx][0])
        actual_label = full_dataset[idx][1]  # 获取真实标签
        print(f"Sample {idx} actual label: {actual_label}")

    # 更新带标记的数据
    newly_labelled_count = len(new_indices)  # 所有选择的样本都被标记
    labelled_indices.extend(new_indices)  # 添加新样本索引
    labelled_indices = list(set(labelled_indices))  # 去重
    labelled_dataset = Subset(full_dataset, labelled_indices)
    labelled_loader = DataLoader(labelled_dataset, batch_size=16, shuffle=True)

    print(f'Iteration {iteration + 1} - Newly labelled samples: {newly_labelled_count}')
    print(f'Iteration {iteration + 1} - Updated labelled samples: {len(labelled_indices)}')

    # 生成新的变异体
    mutants = [create_mutant(model) for _ in range(3)]

    # 评估变异体性能
    for idx, mutant in enumerate(mutants):
        accuracy, precision, recall, f1, conf_matrix, inference_time = evaluate_mutant(mutant)
        print(f'Iteration {iteration + 1} - Mutant {idx + 1} '
              f'Accuracy: {accuracy:.2f}%, '
              f'Precision: {precision:.2f}, '
              f'Recall: {recall:.2f}, '
              f'F1 Score: {f1:.2f}, '
              f'Average Inference Time: {inference_time:.4f} seconds per batch')
        print(f'Confusion Matrix:\n{conf_matrix}')

    # 重新训练模型
    updated_loader = DataLoader(labelled_dataset, batch_size=16, shuffle=True)


    def train_model(model, data_loader, criterion, optimizer, epochs=10):
        for epoch in range(epochs):
            model.train()
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()



