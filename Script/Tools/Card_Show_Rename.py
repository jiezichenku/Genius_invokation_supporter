import os
from Script.Game_Model.Card_Factory import Card_Factory
from PIL import Image


def crop_image(input_image_path, output_image_path, crop_area):
    image = Image.open(input_image_path)
    cropped_image = image.crop(crop_area)
    cropped_image.save(output_image_path)


crop_area = [492, 486, 1526, 629]
factory = Card_Factory()
input_image = "../../Graphic/Cards_Show_Backup"
output = "../../Graphic/Cards_Show"
cards_show = os.listdir(input_image)
for card in cards_show:
    print(card)
    input_image_path = os.path.join(input_image, card)
    output_image_path = os.path.join(output, card)
    crop_image(input_image_path, output_image_path, crop_area)

