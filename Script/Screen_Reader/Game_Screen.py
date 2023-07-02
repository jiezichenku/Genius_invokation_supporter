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
        # 读取卡牌和屏幕
        graphic = card
        img = cv2.imread(graphic)
        screen = self.image.copy()
        # 转换为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        return self.__compare_card(screen, img, threshold, [])

    def check_card(self, card, size, threshold):
        # 读取卡牌和屏幕
        graphic = card
        img = cv2.imread(graphic)
        screen = self.image.copy()
        # 转换为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        # 调整卡牌大小
        # card_height = img.shape[0]
        card_width = img.shape[1]
        new_height = card_width / size[1] * size[0]
        cut_image = img[0:int(new_height), :]
        resized_image = cv2.resize(cut_image, (size[1], size[0]))
        result = self.template_compare(screen, resized_image, threshold)
        return result
        # return self.__compare_card(screen, img, threshold, [])

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

    def template_compare(self, target, img, t):
        # 使用模板匹配方法 TM_CCOEFF_NORMED
        result = cv2.matchTemplate(target, img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # 获取模板的宽度和高度
        template_height, template_width = img.shape[:2]

        # 如果匹配值大于阈值，认为匹配成功
        threshold = t
        if max_val >= threshold:
            # 计算匹配区域的四角坐标
            top_left = max_loc
            bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
            return top_left[0], top_left[1], bottom_right[0], bottom_right[1]
        else:
            return None

    def recognize_numbers_in_rect(self, points):
        # 提取矩形坐标
        x, y, z, w = points
        # 提取矩形区域
        roi = self.image[w-30:w, z-30:z]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 对图像进行文本识别
        deck_num_1 = cv2.imread("../../Graphic/template/deck_num_1.png", 0)
        deck_num_2 = cv2.imread("../../Graphic/template/deck_num_2.png", 0)
        result_1 = self.template_compare(roi, deck_num_1, 0.7)
        if result_1:
            return 1
        result_2 = self.template_compare(roi, deck_num_2, 0.7)
        if result_2:
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
    path = "..\..\Graphic\Support\Liben.png"
    ret = screen_reader.check_card(path, (150, 90), 0.5)
    print(ret)