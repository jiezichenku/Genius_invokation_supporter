import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms


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

# 预测函数
def predict_card(image_path, model, label_names):
    model.eval()
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_height, img_width))
    img = img / 255.0
    img = transforms.ToTensor()(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        predicted_label = label_names[predicted.item()]
    return predicted_label

# 示例预测
# predicted_card = predict_card('../../Graphic/Train_set', model, label_names)
# print(f'预测的牌面: {predicted_card}')
prediction_loss = []
start_time = time.time()
for filename in label_names:
    # print(filename)
    predicted_card = predict_card(os.path.join('../../Graphic/Train_set', filename), model, label_names)

    if predicted_card == filename:
        print(f'预测准确的牌面: {predicted_card}')
    else:
        prediction_loss.append([filename, predicted_card])
        print(f"false alarm: {filename} to {predicted_card}")
end_time = time.time()
print(f"预测准确率：{1 - len(prediction_loss)/len(label_names)}")
print(f"预测错误数：{len(prediction_loss)}")
loss_str = str(prediction_loss).replace('],', ']\n')
print(f"预测错误的牌：{loss_str}")
print(f"运行时间：{end_time - start_time}，单核inference: {(end_time - start_time)/len(label_names)}")