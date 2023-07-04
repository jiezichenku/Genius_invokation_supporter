import os

from Script.Screen_Reader.Game_Screen import Game_Screen


class Game_Status:
    def __init__(self):
        self.status = "init"
        self.status_list = ["deck_character", "deck_card", "prepare", "gaming", "win", "lose"]
        self.graphic_root = "..\..\Graphic"
        self.graphic_folder = ["Support", "Equipment", "Event", "Gift"]
        self.game_screen = Game_Screen()
        self.card_list = []
        self.enemy_card = []
        self.my_card = []
        self.enemy_card_loc = (1190, 175, 1190 + 216, 175 + 376)
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

    def listen_game_screen(self, card_list, card_loc):
        # 读取初始帧
        previous_frame = self.game_screen.get_game_screenshot(card_loc)

        while True:
            # 读取当前帧
            current_frame = self.game_screen.get_game_screenshot(card_loc)

            # 检测画面是否发生较大的变化
            if self.game_screen.detect_change(previous_frame, current_frame, threshold=50000):
                for card in self.card_list:
                    print(card)
                    ret = self.game_screen.check_card(card, [376, 216], 0.5)
                    if ret:
                        card_list.append(card)
                    print(ret)

            previous_frame = current_frame


if __name__ == '__main__':
    status = Game_Status()
    status.add_card_deck()
