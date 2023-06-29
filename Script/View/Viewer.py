import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPixmap


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 256, 768)

        # 创建垂直布局
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 创建按钮布局
        button_layout = QHBoxLayout()

        # 创建三个按钮
        button1 = QPushButton("我方牌库", self)
        button2 = QPushButton("我方已打出", self)
        button3 = QPushButton("对方已打出", self)

        # 连接按钮点击事件到槽函数
        button1.clicked.connect(self.showGroup1)
        button2.clicked.connect(self.showGroup2)
        button3.clicked.connect(self.showGroup3)

        # 将按钮添加到按钮布局
        button_layout.addWidget(button1)
        button_layout.addWidget(button2)
        button_layout.addWidget(button3)

        # 创建视图布局
        view_layout = QVBoxLayout()

        # 创建并添加图片 QLabel
        for i in range(1):  # 假设有5张图片
            image_label = QLabel()
            image_label.setFixedSize(256, 64)
            # 设置图片路径
            image_path = "E:\GitHub\GenshinCard\Graphic\Support\Liyue_Harbor_Wharf.png"  # 假设图片路径为 image0.png, image1.png, ...
            pixmap = QPixmap(image_path)
            image_label.setPixmap(pixmap)
            view_layout.addWidget(image_label)

        # 将按钮布局和视图布局添加到垂直布局
        layout.addLayout(button_layout)
        layout.addLayout(view_layout)

    def showGroup1(self):
        self.clearViewLayout()
        for i in range(3):
            image_label = QLabel()
            image_label.setFixedSize(256, 64)
            image_path = f"group1_image{i}.png"
            pixmap = QPixmap(image_path)
            image_label.setPixmap(pixmap)
            self.view_layout.addWidget(image_label)

    def showGroup2(self):
        self.clearViewLayout()
        for i in range(4):
            image_label = QLabel()
            image_label.setFixedSize(256, 64)
            image_path = f"group2_image{i}.png"
            pixmap = QPixmap(image_path)
            image_label.setPixmap(pixmap)
            self.view_layout.addWidget(image_label)

    def showGroup3(self):
        self.clearViewLayout()
        for i in range(2):
            image_label = QLabel()
            image_label.setFixedSize(256, 64)
            image_path = f"group3_image{i}.png"
            pixmap = QPixmap(image_path)
            image_label.setPixmap(pixmap)
            self.view_layout.addWidget(image_label)

    def clearViewLayout(self):
        while self.view_layout.count() > 0:
            item = self.view_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
