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
        for folder in self.graphic_folder:
            folder_path = os.path.join(self.graphic_root, folder)
            for file in os.listdir(folder_path):
                self.card_list.append(os.path.join(folder_path, file))

    def add_card_deck(self):
        self.game_screen.get_game_screenshot()
        self.game_screen.resize_screen_deck()
        card_deck = {}
        for card in self.card_list:
            print(card)
            ret = self.game_screen.check_card(card, [150, 90], 0.5)
            print(ret)
            if ret:
                num = self.game_screen.recognize_numbers_in_rect(ret)
                if num > 0:
                    card_deck[card.split("\\")[-1]] = num
        print(card_deck)
        print("card_read: %s" % len(card_deck))
        return card_deck


if __name__ == '__main__':
    status = Game_Status()
    status.add_card_deck()
