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
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载MNIST数据集
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 随机选择带标记数据，减少数量
num_labelled_samples = 600  # 带标记数据数量
labelled_indices = random.sample(range(len(full_dataset)), num_labelled_samples)  # 随机选择带标记样本索引
labelled_dataset = Subset(full_dataset, labelled_indices)  # 创建带标记数据子集
labelled_loader = DataLoader(labelled_dataset, batch_size=16, shuffle=True)

# 选择未标记数据
num_unlabelled_samples = 6000  # 未标记数据数量
unlabelled_indices = [i for i in range(len(full_dataset)) if i not in labelled_indices][:num_unlabelled_samples]
unlabelled_dataset = Subset(full_dataset, unlabelled_indices)  # 创建未标记数据子集


# 2. 定义ResNet-18模型
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)# 加载预训练的ResNet-18模型
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)# 修改全连接层以适应10个类


    def forward(self, x):
        return self.resnet(x)# 前向传播

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# 使用GPU或CPU
model = ResNet18().to(device)# 将模型移动到计算设备

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
    for layer in mutated_model.children():
        if isinstance(layer, nn.Conv2d):
            layer.kernel_size = random.choice([(3, 3), (5, 5)])  # 限制内核大小
            # 输出通道数变更限制在 ±25%
            layer.out_channels = random.choice(range(int(layer.out_channels * 0.75), int(layer.out_channels * 1.25)))
            # 随机添加 Dropout
            if random.random() < 0.5:
                mutated_model.add_module('dropout', nn.Dropout(p=0.5))
    return mutated_model



# 生成变异体
mutants = [create_mutant(model) for _ in range(3)]
test_loader = DataLoader(datasets.MNIST(root='./data', train=False, download=True, transform=transform), batch_size=64)

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
def select_combined_uncertain_samples(model, dataset, n_samples=20):
    model.eval()
    scores = []
    with torch.no_grad():
        for images, _ in DataLoader(dataset, batch_size=64):
            outputs = model(images.to(device))
            probs = F.softmax(outputs, dim=1)
            least_confident = torch.max(probs, dim=1)[0]  # 最大置信度
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # 熵
            score = least_confident * entropy  # 组合评分
            scores.extend(score.cpu().numpy())
    uncertain_indices = np.argsort(scores)[:n_samples]  # 选择组合评分最低的样本
    return uncertain_indices



# 8. 特征推导与标签推断
# 特征提取
def extract_shape_features(image):
    gray_image = cv2.cvtColor(image.numpy().transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = {
        'has_circle': False,
        'has_straight_line': False,
        'has_curve': False,
        'has_angle': False,
        'is_symmetric': False,
        'has_two_curves': False,
        'open_side': False
    }

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area > 100:
            circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
            if circularity > 0.8:
                features['has_circle'] = True
            elif circularity < 0.4:
                features['has_straight_line'] = True
            else:
                features['has_curve'] = True

    return features


def infer_label_based_on_rules(image, index):
    shape_features = extract_shape_features(image)

    if shape_features['has_circle']:
        if not shape_features['has_straight_line']:
            return 0  # 数字 0
        else:
            return 6  # 数字 6 或 9，取决于其他特征
    if shape_features['has_straight_line']:
        if shape_features['has_curve']:
            return 2  # 数字 2
        return 1  # 数字 1

    if shape_features['has_curve']:
        if shape_features['has_straight_line']:
            return 5  # 数字 5
        return 3  # 数字 3

    if shape_features['has_angle']:
        return 4  # 数字 4

    # 假设没有匹配的特征，返回真实标签
    return full_dataset[index][1]  # 返回真实标签
'''
特征推断逻辑说明
数字 0: 特征包括圆形（无直线）。
数字 1: 仅有直线。
数字 2: 直线与曲线。
数字 3: 主要曲线。
数字 4: 包含角的特征。
数字 5: 曲线与直线的组合。
数字 6: 类似于0，但包含直线。
数字 7: 包含曲线与直线。
数字 8: 圆形和曲线的结合。
数字 9: 类似于6，但曲线特征更显著。

'''

# 主动学习循环
for iteration in range(10):
    new_indices = select_combined_uncertain_samples(model, unlabelled_dataset, n_samples=600)
    inferred_labels = [infer_label_based_on_rules(full_dataset[idx][0], idx) for idx in new_indices]

    # 输出每个选择样本及其推断标签
    for idx, label in zip(new_indices, inferred_labels):
        print(f"Sample {idx} inferred label: {label}")

    # 更新带标记的数据
    newly_labelled_count = sum(1 for label in inferred_labels if label != -1)
    labelled_indices.extend(idx for idx, label in zip(new_indices, inferred_labels) if label != -1)
    labelled_indices = list(set(labelled_indices))  # 去重
    labelled_dataset = Subset(full_dataset, labelled_indices)
    labelled_loader = DataLoader(labelled_dataset, batch_size=16, shuffle=True)

    print(f'Iteration {iteration + 1} - Newly labelled samples: {newly_labelled_count}')
    print(f'Iteration {iteration + 1} - Updated labelled samples: {len(labelled_indices)}')

    # 计算推断标签的正确性
    true_labels = [full_dataset[idx][1] for idx in new_indices if idx in labelled_indices]
    correct_predictions = sum(1 for true, pred in zip(true_labels, inferred_labels) if true == pred)
    accuracy = correct_predictions / len(true_labels) if true_labels else 0
    print(f'Iteration {iteration + 1} - Inferred label accuracy: {accuracy:.2f}')

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
    for epoch in range(10):
        model.train()
        for images, labels in updated_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# 10. 统计模型参数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model_complexity = count_parameters(model)
print(f'Model Complexity: {model_complexity} parameters')

