import sys
import cv2
import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QStackedLayout, QLabel


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.hide_list = []
        self.card_list = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Main Window")
        self.setFixedSize(300, 800)
        self.top_img("image.png")

        # 创建按钮和栈布局
        button1 = QPushButton("Page 1")
        button2 = QPushButton("Page 2")
        button3 = QPushButton("Page 3")
        button_restore = QPushButton("恢复")

        self.stackedLayout = QStackedLayout()
        self.stackedLayout.addWidget(self.createPage(1, self.card_list))
        # self.stackedLayout.addWidget(self.createPage(2))
        # self.stackedLayout.addWidget(self.createPage(3))

        # 创建按钮布局
        buttonLayout_top = QHBoxLayout()
        buttonLayout_top.addWidget(button1)
        buttonLayout_top.addWidget(button2)
        buttonLayout_top.addWidget(button3)

        buttonLayout_bottom = QHBoxLayout()
        buttonLayout_bottom.addWidget(button_restore)
        # 创建主布局，并将上面的控件添加到主布局中
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.label)
        mainLayout.addLayout(buttonLayout_top)
        mainLayout.addLayout(self.stackedLayout)
        mainLayout.addLayout(buttonLayout_bottom)

        self.setLayout(mainLayout)

        # 连接按钮的点击事件，切换栈布局页面
        button1.clicked.connect(lambda: self.stackedLayout.setCurrentIndex(0))
        button2.clicked.connect(lambda: self.stackedLayout.setCurrentIndex(1))
        button3.clicked.connect(lambda: self.stackedLayout.setCurrentIndex(2))
        button_restore.clicked.connect(self.restoreButton)

    def top_img(self, img):
        # 创建顶部的图片
        pixmap = QPixmap(img).scaled(200, 100)
        self.label = QLabel(self)
        self.label.setPixmap(pixmap)
        self.label.setAlignment(Qt.AlignHCenter)

    def createPage(self, pageNum, card_list):
        # 读取卡牌图片
        cards_show_url = "../../Graphic/Cards_Show"
        cards_show = os.listdir(cards_show_url)
        # 创建按钮控件
        pageWidget = QWidget()
        layout = QVBoxLayout(pageWidget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        for i in range(18):
            url = "%s/%s" % (cards_show_url, cards_show[i + 18 * pageNum])
            img = QImage(url)
            width = 280
            height = int(width / img.width() * img.height())
            button = QPushButton()
            button.setFixedSize(width, height)
            layout.addWidget(button)
            layout.setAlignment(Qt.AlignHCenter)
            button.setStyleSheet("border-image:url(%s)" % url)
            button.clicked.connect(self.onButtonClick)

        return pageWidget

    def onButtonClick(self):
        button = self.sender()  # 获取触发信号的按钮
        button.hide()  # 隐藏按钮
        self.hide_list.append(button)

    def restoreButton(self):
        button = self.hide_list.pop()
        button.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
