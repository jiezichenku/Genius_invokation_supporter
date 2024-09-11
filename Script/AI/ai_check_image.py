import os

import cv2
import numpy as np
import torch
from torchvision import transforms

from ai_train_model import CNNModel

image_folder = '../../Graphic/Train_set'
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
# Load the model
num_classes = len(label_names)
model = CNNModel(num_classes)
model.load_state_dict(torch.load('card_classifier_model.pth'))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor()
])


def predict_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_height, img_width))
    img = transform(img).unsqueeze(0)  # Add batch dimension
    img = img.to(next(model.parameters()).device)

    # Predict
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    predicted_label = label_names[predicted.item()]
    return predicted_label


if __name__ == "__main__":
    for img in os.listdir(image_folder):
        image_path = os.path.join(image_folder, img)
        predicted_label = predict_image(image_path)
        if predicted_label == img:
            pass
            print(f"{img} predicted correct")
        else:
            print(f"{img} predicted incorrect, result {predicted_label}")
