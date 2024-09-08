import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 设置图像尺寸
img_height, img_width = 64, 64

# 读取数据
def load_images_from_folder(folder):
    images = []
    labels = []
    label_names = os.listdir(folder)
    for filename in label_names:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_height, img_width))
        images.append(img)
        labels.append(label_names.index(filename))
    return np.array(images), np.array(labels), label_names

# 加载图像和标签
images, labels, label_names = load_images_from_folder('../../Graphic/Train_set')

# 归一化
images = images / 255.0

# 将标签转换为one-hot编码
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# 切分数据集
X_train, y_train = images, labels
# X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 数据增强和转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((img_height, img_width), scale=(0.9, 1.0)),
    transforms.ToTensor()
])

# 自定义Dataset
class CardDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 创建DataLoader
train_dataset = CardDataset(X_train, y_train, transform=transform)
# test_dataset = CardDataset(X_test, y_test, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# 定义模型
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型
num_classes = len(label_names)
model = CNNModel(num_classes)
model.load_state_dict(torch.load('card_classifier_model.pth'))
model.eval()
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型
# epochs = 1000
# for epoch in range(epochs):
#     model.train()
#     for images, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, torch.max(labels, 1)[1])
#         loss.backward()
#         optimizer.step()
#
#     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        print(predicted)
        correct += (predicted == torch.max(labels, 1)[1]).sum().item()

accuracy = correct / total
print(f'测试集准确率: {accuracy:.2f}')

# 保存模型
torch.save(model.state_dict(), 'card_classifier_model.pth')
