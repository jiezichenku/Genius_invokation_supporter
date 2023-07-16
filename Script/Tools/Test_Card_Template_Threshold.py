from Script.Game_Model.Card_Factory import Card_Factory
from Script.Screen_Reader.Image_Recognize import Image_Recognize
import time


start_time = time.time()
img_recognize = Image_Recognize()
factory = Card_Factory()

card_list = factory.cards_by_id
x = 163
y = 83
w = 50
h = 50
region = [x, y, w, h]
# 预处理图片
for card in card_list:
    print(card.ch)
    img = img_recognize.get_img(card.full_img)
    img = img_recognize.resize(img, (216, 376))
    result = img_recognize.check_card(img, card_list, region=None, compare_size=region)
    print(result[0].ch, result[1])
compare_time = time.time()
print("比较时间：%s" % (compare_time - start_time))
