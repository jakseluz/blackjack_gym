from enum import Enum

MAX_POINTS = 21
DEALER_LIMIT = 17


class Actions(Enum):
    HIT = 0
    STAND = 1
    DOUBLE_DOWN = 2
    SPLIT = 3
    INSURANCE = 4


class Game_version(Enum):
    """Enum for the version of the game.
    AMERICAN: The dealer receives two cards at the beginning of the game and the player receives two cards. The player can double down on any two initial cards and can split pairs at the beginning. The player can also take insurance if the dealer's visible card is an ace.
    EUROPEAN: The dealer receives one card at the beginning of the game and the player receives two cards. The player can only double down on 9, 10, or 11 and can only split pairs of 8s or Aces at the beginning. The player cannot take insurance.
    """

    AMERICAN = 0
    EUROPEAN = 1


class Casino_ace(Enum):
    STAND_ON_ALL_17S = 0
    HIT_SOFT_17 = 1


class Reward(Enum):
    blackjack32 = 1.5
    blackjack65 = 1.2


# TODO
class SurrenderRule(Enum):
    """
    Enum for the surrender rule.
    NOT_AVAILABLE: Surrender is not available.
    LATE_SURRENDER: Surrender is available only after the dealer checks blackjack.
    EARLY_SURRENDER: Player gives a half payout before dealer checks blackjack.
    """

    NOT_AVAILABLE = 0
    LATE_SURRENDER = 1
    EARLY_SURRENDER = 2
