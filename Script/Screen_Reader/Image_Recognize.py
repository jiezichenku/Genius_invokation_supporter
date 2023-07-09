import cv2

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
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def resize(self, img, size):
        if img is None:
            raise Exception("Image_Recognize.resize函数报错：未检测到图片")
        return cv2.resize(img, (size[0], size[1]))

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