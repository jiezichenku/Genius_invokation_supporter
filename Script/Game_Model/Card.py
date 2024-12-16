class Card:
    def __init__(self, card_id=None, en="", ch="", full_img="", show_img="", tag=""):
        self.id = card_id
        self.en = en
        self.ch = ch
        self.full_img = full_img
        self.show_img = show_img
        self.tag = tag

    def __str__(self):
        return "[ID: %s, 英文名：%s, 中文名：%s]\n完整图：%s\n缩略图：%s" % \
               (self.id, self.en, self.ch, self.full_img, self.show_img)
