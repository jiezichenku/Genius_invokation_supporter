import cv2
import numpy as np


class Image_Recognize:
    def __init__(self):
        pass

    def show_img(self, img):
        cv2.imshow('IMG', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def to_single_channel(self, img):
        if img is None:
            raise Exception("Image_Recognize.to_single_channel函数报错：未检测到图片")
        if len(img.shape) < 3:
            return img
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img

    def resize(self, img, size):
        if img is None:
            raise Exception("Image_Recognize.resize函数报错：未检测到图片")
        return cv2.resize(img, (size[0], size[1]))

    def template_compare(self, target, img, t):
        # 使用模板匹配方法 TM_CCOEFF_NORMED
        target = self.to_single_channel(target)
        img = self.to_single_channel(img)
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

    def check_card(self, img, card, threshold, region=None):
        # 转换为灰度图
        card = cv2.imread(card)
        img = self.to_single_channel(img)
        card = self.to_single_channel(card)
        if region:
            img = img[region[0]:region[1], region[2]:region[3]]
        # 调整卡牌大小
        size = img.shape
        card_height = card.shape[0]
        card_width = card_height / size[0] * size[1]
        cut_image = card[0:int(card_height), 0:int(card_width)]
        resized_image = self.resize(cut_image, (size[1], size[0]))
        result = self.template_compare(img, resized_image, threshold)
        return result

    def recognize_numbers_in_rect(self, img, points):
        img = self.to_single_channel(img)
        # 提取矩形坐标
        x, y, z, w = points
        # 提取矩形区域
        roi = img[w-30:w, z-30:z]
        roi = self.to_single_channel(roi)
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

    def detect_change(self, previous_frame, current_frame, threshold=30):
        # 计算帧间差分的绝对值
        frame_diff = cv2.absdiff(previous_frame, current_frame)
        # 计算像素级别的变化总和
        diff_sum = np.sum(frame_diff)
        # 检查是否超过阈值
        if diff_sum > threshold:
            return diff_sum
        else:
            return False
