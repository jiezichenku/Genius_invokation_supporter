import numpy as np
import pyautogui
import time
import pytesseract
import os
import easyocr
from PIL import Image
import cv2


class Game_Screen:
    def __init__(self):
        self.image = None

    def get_game_screenshot(self, left, top, width, height):
        game_window_title = "Genshin Impact"
        game_window = pyautogui.getWindowsWithTitle(game_window_title)[0]
        if game_window is not None:
            game_window.activate()
            time.sleep(1)
            self.image = pyautogui.screenshot(region=(game_window.left + left, game_window.top + top,
                                                      width, height))
        return self.image

    def check_card_exist(self, screen, card, threshold):
        # 图片文件夹路径和游戏画面截图路径
        graphic = card  # 卡牌图片文件夹路径
        # target = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        # 遍历图片文件夹中的所有图片
        img = cv2.imread(graphic, 0)
        self.image = cv2.imread(screen, 0)
        return self.__compare_card(self.image, img, threshold, [])

    def __compare_card(self, target, img, threshold, ret):
        # 初始化FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # 检测和计算关键点和描述符
        sift = cv2.SIFT_create()
        kp_img, des_img = sift.detectAndCompute(img, None)
        kp_target, des_target = sift.detectAndCompute(target, None)

        # 使用FLANN匹配器进行描述符匹配
        matches = flann.knnMatch(des_img, des_target, k=2)

        # 应用比率测试，筛选出较好的匹配
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)

        # 获取模板在屏幕截图中的位置
        locations = []
        for match in good_matches:
            x, y = kp_target[match.trainIdx].pt
            locations.append((int(x), int(y)))

        if len(good_matches) > 10:
            # 提取匹配点的坐标
            src_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 使用RANSAC算法计算透视变换矩阵
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # 获取图片的宽度和高度
            img_width, img_height = img.shape[1], img.shape[0]

            # 计算图像四个角经透视变换后的坐标
            pts = np.float32([[0, 0], [0, img_height], [img_width, img_height], [img_width, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # 将匹配到的卡牌涂白
            target_next = cv2.fillPoly(target, [np.int32(dst)], 255)
            ret.append(dst)
            ret = self.__compare_card(target_next, img, threshold, ret)
            return ret

        else:
            pass
            # 显示带有标记的目标图像
            cv2.imshow('Target with Match', target)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return ret

    def check_screen_rect(self, screen, threshold1, threshold2):
        # 读取图像
        image = cv2.imread(screen, 0)  # 0表示以灰度图像方式读取
        # 自适应阈值处理
        adaptive_threshold = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 20)
        _, binary = cv2.threshold(adaptive_threshold, 0, 255, cv2.THRESH_BINARY)
        # 显示结果
        cv2.imshow('Adaptive Threshold', binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 轮廓提取
        contours, _ = cv2.findContours(adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 筛选矩形区域
        rectangles = []
        for contour in contours:
            # 近似多边形为矩形
            epsilon = 0 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 筛选满足条件的矩形区域
            if len(approx) == 4 and cv2.isContourConvex(approx):
                rectangles.append(approx)

        # 绘制矩形区域
        for rect in rectangles:
            x, y, w, h = cv2.boundingRect(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示结果
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def recognize_numbers_in_rect(self, points, distance_threshold):
        rect = cv2.boundingRect(points)
        # 提取矩形坐标
        x, y, w, h = rect

        # 根据距离阈值计算角落的位置
        top_left = [x - distance_threshold, y - distance_threshold]
        bottom_right = [x + w + distance_threshold, y + h + distance_threshold]
        # cv2.namedWindow("Window", cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Window', self.image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # 提取矩形区域
        roi = self.image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        # cv2.imshow('Target with Match', roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 对图像进行文本识别
        pil_image = Image.fromarray(roi)
        reader = easyocr.Reader(['en'])
        results = reader.readtext(pil_image)
        return results

if __name__ == '__main__':
    screen_reader = Game_Screen()
    # screen_reader.get_game_screenshot()
    # time.sleep(1)
    screen = "E:\GitHub\GenshinCard\Graphic\Game\\test.png"
    path = "E:\GitHub\GenshinCard\Graphic\Support\Liyue_Harbor_Wharf.png"

    # for i in [1]:
    #     t1 = i*0.1
    #     t2 = i*0.2
    #     screen_reader.check_screen_rect(screen, t1, t2)
    #     print("threshold: %s+%s" % (t1, t2))
    for i in [5]:
        t = 0.1 * i
        ret = screen_reader.check_card_exist(screen, path, t)
        ret = screen_reader.recognize_numbers_in_rect(ret[0], 5)
        print("threshold: %s, card_num: %s" % (t, ret))


