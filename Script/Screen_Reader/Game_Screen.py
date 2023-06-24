import cv2
import numpy as np
import pyautogui
import time
import os


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

    def check_card_exist(self, card, threshold):
        # 图片文件夹路径和游戏画面截图路径
        graphic = card  # 卡牌图片文件夹路径
        target = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        # 初始化FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 遍历图片文件夹中的所有图片
        img = cv2.imread(graphic, 0)

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


        if len(good_matches) > 10:
            # 提取匹配点的坐标
            src_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            print("src_pts: %s \ndst_pts: %s" % (src_pts, dst_pts))

            # 使用RANSAC算法计算透视变换矩阵
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # 获取图片的宽度和高度
            img_width, img_height = img.shape[1], img.shape[0]

            # 计算图像四个角经透视变换后的坐标
            pts = np.float32([[0, 0], [0, img_height], [img_width, img_height], [img_width, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            return dst

        return None

    def check_screen_status(self, template):
        # 读取游戏界面截图
        game_screen = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
        # 读取模板图像
        template_image = cv2.imread(template)

        # 使用模板匹配算法
        result = cv2.matchTemplate(game_screen, template_image, cv2.TM_CCOEFF_NORMED)

        # 设置匹配阈值
        threshold = 0.8
        # 获取匹配结果中匹配位置的坐标
        locations = np.where(result >= threshold)
        # 绘制矩形框标记匹配位置
        for pt in zip(*locations[::-1]):
            cv2.rectangle(game_screen, pt, (pt[0] + template_image.shape[1], pt[1] + template_image.shape[0]),
                          (0, 255, 0), 2)

        # 显示带有标记的游戏界面
        cv2.imshow('Game Screen', game_screen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    screen_reader = Game_Screen()
    screen_reader.get_game_screenshot()
    time.sleep(1)
    path = "E:\GitHub\GenshinCard\Graphic\Event\I_Havent_Lost_Yet.png"
    num, pos = screen_reader.check_card_exist(path, 0.9)
    print("num: %s, pos:\n%s" % (num, pos))



