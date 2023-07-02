import os
import numpy as np
import pyautogui
import time
import cv2


class Game_Screen:
    def __init__(self):
        self.image_size = [1600, 900]
        self.image = None

    def show_img(self, img):
        cv2.imshow('IMG', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_game_screenshot(self):
        game_window_title = "原神"
        game_window = pyautogui.getWindowsWithTitle(game_window_title)[0]
        # 图片分辨率处理
        if game_window is not None:
            game_window.activate()
            time.sleep(1)
            screenshot = pyautogui.screenshot(region=(game_window.left, game_window.top,
                                                      game_window.width, game_window.height))
            self.image = np.array(screenshot)

    def resize_screen_gaming(self):
        # 将屏幕适配为16:9，不为16:9的上下裁去等量宽度改为16:9

        # 获取原始图像的宽度和高度
        height = self.image.shape[0]
        width = self.image.shape[1]

        # 计算原始图像的长宽比
        aspect_ratio = width / height
        if aspect_ratio == 16 / 9:
            # 如果原始图像的长宽比为16:9，则直接缩放为目标尺寸
            resized_image = cv2.resize(self.image, (self.image_size[0], self.image_size[1]))
        else:
            # 计算裁剪后的宽度
            cropped_height = int(width * (9 / 16))
            crop_start = int((height - cropped_height) / 2)
            crop_end = crop_start + cropped_height

            # 上下裁剪相等的宽度
            cropped_image = self.image[crop_start:crop_end, :]
            self.show_img(cropped_image)

            # 缩放为目标尺寸
            resized_image = cv2.resize(cropped_image, (self.image_size[0], self.image_size[1]))

        self.image = resized_image

    def resize_screen_deck(self):
        # 将屏幕宽度调整为1600，整个屏幕等比例放大
        height = self.image.shape[0]
        width = self.image.shape[1]

        aspect_ratio = width / height
        resized_width = self.image_size[0]
        resized_height = resized_width / aspect_ratio
        resized_image = cv2.resize(self.image, (int(resized_width), int(resized_height)))
        self.image = resized_image

    def check_card_exist(self, card, threshold):
        # 图片文件夹路径和游戏画面截图路径
        graphic = card  # 卡牌图片文件夹路径
        # 遍历图片文件夹中的所有图片
        img = cv2.imread(graphic)
        screen = self.image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        return self.__compare_card(screen, img, threshold, [])

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

        if des_img is None:
            print("des_img: error")
            return ret

        if des_target is None:
            print("des_target: error")
            return ret

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
            target_next = target.copy()
            cv2.fillPoly(target_next, [np.int32(dst)], 255)

            ret.append(dst)
            ret = self.__compare_card(target_next, img, threshold, ret)
            return ret
        return ret

    def template_compare(self, target, img, threshold):
        # 将图片切换为灰度图像
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        # 使用模板匹配方法 TM_CCOEFF_NORMED
        res = cv2.matchTemplate(img, target, cv2.TM_CCOEFF_NORMED)

        # 使用np.where函数检测匹配程度超过阈值的位置
        locations = np.where(res >= threshold)
        return locations

    def check_screen_rect(self, screen, threshold1, threshold2):
        # 读取图像
        image = cv2.imread(screen, 0)  # 0表示以灰度图像方式读取
        # 自适应阈值处理
        adaptive_threshold = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 20)
        _, binary = cv2.threshold(adaptive_threshold, 0, 255, cv2.THRESH_BINARY)
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

    def recognize_numbers_in_rect(self, points):
        rect = cv2.boundingRect(points)
        # 提取矩形坐标
        x, y, w, h = rect
        # 提取矩形区域
        roi = self.image[y:y + h, x:x + w]
        # 对图像进行文本识别
        deck_num_1 = cv2.imread("../../Graphic/template/deck_num_1.png", 0)
        deck_num_2 = cv2.imread("../../Graphic/template/deck_num_2.png", 0)
        result_1 = self.template_compare(roi, deck_num_1, 0.7)
        if len(result_1[0]) > 0:
            return 1
        result_2 = self.template_compare(roi, deck_num_2, 0.7)
        if len(result_2[0]) > 0:
            return 2
        return 0


def all_card(card_path):
    card_ret = []
    for file in os.listdir(card_path):
        card_ret.append(os.path.join(card_path, file))
    return card_ret


if __name__ == '__main__':
    screen_reader = Game_Screen()
    screen_reader.get_game_screenshot()
    screen_reader.resize_screen_deck()
    # time.sleep(1)
    # screen = "E:\GitHub\GenshinCard\Graphic\Game\\deck_card.png"
    # screen_reader.image = cv2.imread(screen, 0)
    root = "E:\GitHub\Genius_invoke\Graphic"
    path = ["Support", "Equipment", "Event", "Gift"]

    # for i in [1]:
    #     t1 = i*0.1
    #     t2 = i*0.2
    #     screen_reader.check_screen_rect(screen, t1, t2)
    #     print("threshold: %s+%s" % (t1, t2))
    deck = {}
    t = 0.5
    for p in path:
        for card in all_card(os.path.join(root, p)):
            print(card)
            ret = screen_reader.check_card_exist(card, t)
            if len(ret) > 0:
                num = screen_reader.recognize_numbers_in_rect(ret[0])
                if num > 0:
                    deck[card.split("\\")[-1]] = num
    print(deck)
