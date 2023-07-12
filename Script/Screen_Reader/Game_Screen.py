import numpy as np
import pyautogui
from Script.Screen_Reader.Image_Recognize import Image_Recognize


class Game_Screen:
    def __init__(self):
        self.image_recognize = Image_Recognize()
        self.image_size = [1600, 900]
        self.image = None

    def get_game_screenshot(self):
        game_window_title = "原神"
        game_window = pyautogui.getWindowsWithTitle(game_window_title)[0]
        # 图片分辨率处理
        if game_window is not None:
            game_window.activate()
            # time.sleep(1)
            region = (game_window.left, game_window.top, game_window.width, game_window.height)
            screenshot = pyautogui.screenshot(region=region)
            self.image = np.array(screenshot)
        else:
            raise Exception("未检测到游戏屏幕")
        self.image = self.image_recognize.to_single_channel(self.image)

    def resize_screen_gaming(self):
        # 将屏幕适配为16:9，不为16:9的上下裁去等量宽度改为16:9
        # 获取原始图像的宽度和高度
        height = self.image.shape[0]
        width = self.image.shape[1]

        # 计算原始图像的长宽比
        aspect_ratio = width / height
        if aspect_ratio == 16 / 9:
            # 如果原始图像的长宽比为16:9，则直接缩放为目标尺寸
            resized_image = self.image_recognize.resize(self.image, self.image_size)
        else:
            # 计算裁剪后的宽度
            cropped_height = int(width * (9 / 16))
            crop_start = int((height - cropped_height) / 2)
            crop_end = crop_start + cropped_height

            # 上下裁剪相等的宽度
            cropped_image = self.image[crop_start:crop_end, :]

            # 缩放为目标尺寸
            resized_image = self.image_recognize.resize(cropped_image, self.image_size)

        self.image = resized_image

    def resize_screen_deck(self):
        # 将屏幕宽度调整为1600，整个屏幕等比例放大
        height = self.image.shape[0]
        width = self.image.shape[1]

        aspect_ratio = width / height
        resized_width = self.image_size[0]
        resized_height = resized_width / aspect_ratio
        resized_image = self.image_recognize.resize(self.image, (int(resized_width), int(resized_height)))
        self.image = resized_image

    def get_screen_area(self, region):
        # 获取屏幕区域并转为灰度图
        screen = self.image.copy()
        screen = screen[region[0]:region[1], region[2]:region[3]]
        return screen
