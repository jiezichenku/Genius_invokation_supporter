import json
from Script.Game_Model.Card import Card
Card_Info_Json = "../../Data/Card/Card_Info.json"
Card_Usage_Json = "../../Data/Card/Card_Usage.json"


class Card_Factory:
    def __init__(self):
        self.cards_by_id = []
        self.cards_by_usage = []

    def cards_by_id(self):
        return self.cards_by_id

    def cards_by_usage(self):
        return self.cards_by_usage

    def get_card_by_cn(self, ch):
        for card in self.cards_by_id:
            if card.ch == ch:
                return card
        return None

    def read_card(self):
        with open(Card_Info_Json, encoding='utf-8') as file:
            data = json.load(file)
            for card_info in data:
                card = Card(card_info["id"],
                            card_info["en"],
                            card_info["ch"],
                            card_info["full_img"],
                            card_info["show_img"])
                self.cards_by_id.append(card)
        print(len(self.cards_by_id))

        with open(Card_Usage_Json, encoding='utf-8') as file:
            data = json.load(file)
            for key in data.keys():
                if len(self.cards_by_usage) == 0:
                    self.cards_by_usage.append(self.get_card_by_cn(key))
                else:
                    inserted = False
                    for i in range(len(self.cards_by_usage)):
                        if data[key] > data[self.cards_by_usage[i].ch]:
                            self.cards_by_usage.insert(i, self.get_card_by_cn(key))  # 在正确位置插入数字
                            inserted = True
                            break
                    if not inserted:
                        self.cards_by_usage.append(self.get_card_by_cn(key))


if __name__ == '__main__':
    factory = Card_Factory()
    factory.read_card()
