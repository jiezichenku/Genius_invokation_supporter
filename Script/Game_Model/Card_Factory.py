import json
import os
Card_Info_Json = "../../Data/Card/Card_Info.json"
Card_Usage_Json = "../../Data/Card/Card_Usage.json"


class Card_Factory:
    def __init__(self):
        self.card_by_id = []
        self.card_by_usage = []

    def card_by_id(self):
        return self.card_by_id

    def card_by_usage(self):
        return self.card_by_usage

    def read_card(self):
        with open('data.json') as file:
            data = json.load(file)