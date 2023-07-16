import json
from Script.Game_Model.Card import Card
Card_Info_Json = "../../Data/Card/Card_Info.json"
Card_Usage_Json = "../../Data/Card/Card_Usage.json"
full_path = "../../Graphic/Cards_Full"
show_path = "../../Graphic/Cards_Show"

# 读取卡牌信息
with open(Card_Info_Json, encoding='utf-8') as file:
    data = json.load(file)
    for card_info in data:
        card = Card(card_info["id"],
                    card_info["en"],
                    card_info["ch"],
                    card_info["full_img"],
                    card_info["show_img"])
        card_info["full_img"] = "%s/%s.png" % (full_path, card_info["en"])
        card_info["show_img"] = "%s/%s.png" % (show_path, card_info["ch"])

        print(card)

# 写入卡牌信息
print(data)
with open(Card_Info_Json, "w", encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)
    print("写入完成")