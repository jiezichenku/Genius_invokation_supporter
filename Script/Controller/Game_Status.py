import os

from Script.Screen_Reader.Game_Screen import Game_Screen


class Game_Status:
    def __init__(self):
        self.status = "init"
        self.status_list = ["deck_character", "deck_card", "prepare", "gaming", "win", "lose"]
        self.graphic_path = "../../Graphic/Game"
        self.game_screen = Game_Screen()

    def check_status(self):
        status_changed = False
        for status in self.status_list:
            graphic = os.path.join(self.graphic_path, status)
            if self.game_screen.check_screen_status(graphic):
                self.status = status
                status_changed = True
        if not status_changed:
            self.status = "init"
