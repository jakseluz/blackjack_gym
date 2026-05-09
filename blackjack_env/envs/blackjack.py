import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum

MAX_POINTS = 21


class Actions(Enum):
    HIT = 0
    STAND = 1
    DOUBLE_DOWN = 2
    SPLIT = 3
    INSURANCE = 4


class BlackjackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        # 1. Zdefiniuj przestrzenie (akcje i obserwacje)
        # super(BlackjackEnv, self).__init__()
        self.observation_space = spaces.Dict(
            {
                "player_points": spaces.Discrete(21),
                "not_used_ace": spaces.Discrete(4),
                "dealer_card": spaces.Discrete(10),
            }
        )

        self.action_space = spaces.Discrete(5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        # 2. Przywróć stan początkowy gry
        super().reset(seed=seed)

        self.deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4
        picked = self._get_cards(4)
        self.player_cards = picked[:2]
        self.dealer_cards = picked[2:]

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        # 3. Wykonaj ruch, oblicz nagrodę, sprawdź czy koniec
        if action == Actions.HIT.value:
            self.player_cards.append(self._get_cards(1))
        elif action == Actions.STAND.value:
            pass
        elif action == Actions.DOUBLE_DOWN.value:
            pass
        elif action == Actions.SPLIT.value:
            pass
        elif action == Actions.INSURANCE.value:
            pass

        reward = 0.0
        if sum(self.player_cards) >= MAX_POINTS or sum(self.dealer_cards) >= MAX_POINTS:
            terminated = True
        else:
            terminated = False
        observation = self._get_observation()
        info = {"action_mask": self._get_action_mask()}
        truncated = False
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def _get_cards(self, num) -> int | list:
        idxs = self.np_random.choice(len(self.deck), size=num, replace=False)
        picked = [self.deck[i] for i in idxs]
        for v in picked:
            self.deck.remove(v)
        return picked[0] if len(picked) == 1 else picked

    def _get_observation(self) -> dict:
        player_usable_aces = self.player_cards.count(11)
        points = sum(self.player_cards)
        if points > MAX_POINTS:
            if player_usable_aces > 0:
                points -= 10
                player_usable_aces -= 1
                self.player_cards[self.player_cards.index(11)] = 1
        return {"player_points": points, "not_used_ace": player_usable_aces, "dealer_card": self.dealer_cards[0]}

    def _get_action_mask(self) -> np.ndarray:
        mask = np.ones(5, dtype=np.int8)
        return mask
