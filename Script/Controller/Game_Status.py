import datetime
import os
import time

import cv2
from Script.Screen_Reader.Game_Screen import Game_Screen
from Script.Game_Model.Card_Factory import Card_Factory

class Game_Status:
    def __init__(self):
        self.status = "init"
        self.status_list = ["deck_character", "deck_card", "prepare", "gaming", "win", "lose"]
        self.graphic_root = "..\..\Graphic"
        self.graphic_folder = ["Support", "Equipment", "Event", "Gift"]
        self.game_screen = Game_Screen()
        self.card_factory = Card_Factory()
        self.card_factory.read_card()
        self.card_list = []
        self.enemy_card = []
        self.my_card = []
        self.enemy_card_loc = (175, 175 + 376, 1190, 1190 + 216)
        self.game_status_bar = (600, 1000, 440, 490)
        self.my_card_loc = ()
        self.enemy_card_img = None
        self.my_card_img = None
        for folder in self.graphic_folder:
            folder_path = os.path.join(self.graphic_root, folder)
            for file in os.listdir(folder_path):
                self.card_list.append(os.path.join(folder_path, file))

    def add_card_deck(self):
        self.game_screen.get_game_screenshot()
        self.game_screen.resize_screen_deck()
        card_deck = {}
        card_location = {}
        for card in self.card_list:
            print(card)
            ret = self.game_screen.check_card(card, [150, 90], 0.5)
            print(ret)
            if ret:
                num = self.game_screen.recognize_numbers_in_rect(ret)
                if num > 0:
                    card_deck[card.split("\\")[-1]] = num
                    card_location[card.split("\\")[-1]] = ret
        print(card_deck)
        print("card_read: %s" % len(card_deck.keys()))
        return card_deck

    def listen_game_screen(self, card_loc):
        # 读取初始帧
        self.game_screen.get_game_screenshot()
        self.game_screen.resize_screen_gaming()
        previous_frame = self.game_screen.image.copy()
        previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        previous_frame = previous_frame[card_loc[0]:card_loc[1], card_loc[2]:card_loc[3]]
        cards_img = []
        diff_sums = []
        while True:
            # 读取当前帧
            self.game_screen.get_game_screenshot()
            self.game_screen.resize_screen_gaming()
            current_frame = self.game_screen.image.copy()
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            current_frame = current_frame[card_loc[0]:card_loc[1], card_loc[2]:card_loc[3]]
            # 检测画面是否发生较大的变化
            diff_sum = self.game_screen.detect_change(previous_frame, current_frame, threshold=1000000)
            if diff_sum is not False:
                print(diff_sum)
                cards_img.append(current_frame)
                diff_sums.append(diff_sum)
                # for card in self.card_list:
                #     ret = self.game_screen.check_card(card, [376, 216], 0.9, card_loc)
                #     if ret:
                #         card_list.append(card)
                #         print(card)
                #         break
            if len(cards_img) > 100:
                break
            previous_frame = current_frame
        for i in range(len(cards_img)):
            cv2.imwrite("%s.png" % diff_sums[i], cards_img[i])


if __name__ == '__main__':
    status = Game_Status()
    status.listen_game_screen(status.enemy_card, status.enemy_card_loc)
    # status.add_card_deck()
