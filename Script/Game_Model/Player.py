from Script.Game_Model.Character import Character
from Script.Game_Model.Deck import Deck
from Script.Game_Model.Dices import Dices
from Script.Game_Model.Graveyard import Graveyard
from Script.Game_Model.Hand import Hand
from Script.Game_Model.Minion import Minion
from Script.Game_Model.Support import Support


class Player:
    def __init__(self):
        self.deck = Deck()
        self.hand = Hand()
        self.graveyard = Graveyard()
        self.support = Support()
        self.minion = Minion()
        self.dices = Dices()
        self.character = Character()
