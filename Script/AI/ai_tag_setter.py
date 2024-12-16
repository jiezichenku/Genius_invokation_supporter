import os
import json

Card_Info = "../../Data/characters.json"
Tags = '../../Graphic/Train_set'


def add_tag_to_info():
    with open(Card_Info, encoding='utf-8') as f:
        data = json.load(f)
        print(len(data))
    card_tags = os.listdir(Tags)
    print(len(card_tags))
    for tag in card_tags:
        for card_info in data:
            if card_info["englishName"].replace(" ", "_") in tag:
                print(tag, card_info["englishName"])
                card_info["tag"] = tag
    with open(Card_Info, 'w', encoding='gbk') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    add_tag_to_info()
