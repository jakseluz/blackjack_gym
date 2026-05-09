import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum


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
                "player_cards": spaces.Box(1, 10, shape=(2,), dtype=int),
                "dealer_card": spaces.Discrete(10),
            }
        )

        self.action_space = spaces.Discrete(5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # 2. Przywróć stan początkowy gry
        super().reset(seed=seed)
        observation = None
        info = {}
        return observation, info

    def step(self, action):
        # 3. Wykonaj ruch, oblicz nagrodę, sprawdź czy koniec
        observation = None
        reward = 0.0
        terminated = False
        truncated = False
        info = {"action_mask": self._get_action_mask()}
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def _get_action_mask(self):
        mask = np.ones(5, dtype=np.int8)
        return mask
