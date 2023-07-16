import cv2
import numpy as np


class Image_Recognize:
    def __init__(self):
        pass

    def get_img(self, path):
        return cv2.imread(path)

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

    def clip(self, img, x, y, w, h):
        # x, y为左上角坐标, w, h为宽高
        return img[x:x+w, y:y+h]

    def template_compare(self, target, img):
        # 使用模板匹配方法 TM_CCOEFF_NORMED
        target = self.to_single_channel(target)
        img = self.to_single_channel(img)
        result = cv2.matchTemplate(target, img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val

    def check_card(self, img, card_list, region=None, compare_size=None):
        # 处理屏幕截图
        img = self.to_single_channel(img)
        if region:
            img = img[region[0]:region[1], region[2]:region[3]]
        size = img.shape
        if compare_size:
            img = self.clip(img, compare_size[0], compare_size[1], compare_size[2], compare_size[3])
        # 遍历卡牌列表，获取匹配度最高的卡牌
        compare_res = 0
        current_card = ""
        for card in card_list:
            card_image = cv2.imread(card.full_img)
            card_image = self.to_single_channel(card_image)
            # 调整卡牌大小
            # card_height = card_image.shape[0]
            # card_width = card_height / size[0] * size[1]
            # cut_image = card_image[0:int(card_height), 0:int(card_width)]
            resized_image = self.resize(card_image, (size[1], size[0]))
            if compare_size:
                resized_image = self.clip(resized_image, compare_size[0], compare_size[1], compare_size[2], compare_size[3])
            result = self.template_compare(img, resized_image)
            if result > compare_res:
                compare_res = result
                current_card = card
        return current_card, compare_res

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
        result_1 = self.template_compare(roi, deck_num_1)
        result_2 = self.template_compare(roi, deck_num_2)
        if result_1 < 0.7 and result_2 < 0.7:
            return 0
        if result_1 > result_2:
            return 1
        if result_1 < result_2:
            return 2

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
