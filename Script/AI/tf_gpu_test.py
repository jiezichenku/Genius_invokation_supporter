import os
import time
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 设置图像尺寸
MAX_WIDTH = 420
MAX_HEIGHT = 720
img_height, img_width = 512, 512
train_set = "../../Graphic/Train_set"
test_set = "../../Graphic/Test_set"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


# 训练数据
def train_model(train_set, save_path):
    # 加载图像和标签
    images, labels, label_names = load_images_from_folder(train_set)

    # 归一化
    images = images / 255.0

    # 将标签转换为one-hot编码
    labels = to_categorical(labels, num_classes=len(label_names))

    # 准备数据集
    X_train = images
    y_train = labels
    X_test = images
    y_test = labels
    # X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # 数据增强
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # 创建模型
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(label_names), activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=500, validation_data=(X_test, y_test))

    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'测试集准确率: {accuracy:.2f}')

    # 保存模型
    model.save(save_path)
    return model


# 预测函数
def predict_card(test_set, image_path, model, label_names):
    img_path = os.path.join(test_set, image_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_height, img_width))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_label = label_names[np.argmax(prediction)]
    return predicted_label


# 示例预测
def test(model, test_set):
    prediction_loss = []
    start_time = time.time()
    label_names = os.listdir(test_set)
    for filename in label_names:
        if not filename.endswith(".png"):
            continue
        predicted_card = predict_card(test_set, filename, model, label_names)

        if predicted_card == filename:
            print(f'预测准确的牌面: {predicted_card}')
        else:
            prediction_loss.append([filename, predicted_card])
            print(f"false alarm: {filename} to {predicted_card}")

    end_time = time.time()
    loss_str = str(prediction_loss).replace('],', ']\n')
    print(f"预测准确率：{1 - len(prediction_loss) / len(label_names)}")
    print(f"预测错误的牌：{loss_str}")
    print(f"预测错误数：{len(prediction_loss)}")
    print(f"运行时间：{end_time - start_time}，单核inference: {(end_time - start_time)/len(label_names)}")


if __name__ == "__main__":
    # model = train_model(train_set, 'temp.h5')
    model = load_model('temp.h5')
    test(model, train_set)
