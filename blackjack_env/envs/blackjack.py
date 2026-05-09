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
        """Initializes the Blackjack environment. The observation space consists of the player's points, the number of not used aces, and the dealer's visible card. The action space consists of 5 actions: hit, stand, double down, split, and insurance.
        Args:
            render_mode (str, optional): The mode to render the environment. Defaults to None.
        """
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
        """Resets the environment to an initial state and returns the initial observation and info.
        Args:
            seed (int, optional): The seed for the random number generator (self.np_random). Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.
        Returns:
            tuple[dict, dict]: A tuple containing the initial observation and info.
        """
        super().reset(seed=seed)

        self.deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4
        picked = self._get_cards(4)
        self.player_cards = picked[:2]
        self.dealer_cards = picked[2:]

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """Performs the given action in the environment and returns the resulting observation, reward, terminated, truncated, and info.
        Args:
            action (int): The action to perform, represented as an integer corresponding to the Actions enum.
        Returns:
            tuple[dict, float, bool, bool, dict]: A tuple containing the resulting observation, reward, terminated, truncated, and info.
        """
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
        """Renders the current state of the environment. If the render mode is "human", it will display a window with the current state. If the render mode is "rgb_array", it will return an RGB array representing the current state."""
        pass

    def _get_cards(self, num) -> int | list:
        """Draws a specified number of cards from the deck and returns them. The drawn cards are removed from the deck.
        Args:
            num (int): The number of cards to draw.
        Returns:
            int | list: A single card if num is 1, otherwise a list of drawn cards.
        """
        idxs = self.np_random.choice(len(self.deck), size=num, replace=False)
        picked = [self.deck[i] for i in idxs]
        for v in picked:
            self.deck.remove(v)
        return picked[0] if len(picked) == 1 else picked

    def _dealer_move(self):
        """Performs the dealer's move according to the rules of Blackjack. The dealer will keep drawing cards until their points are 17 or higher."""
        if sum(self.dealer_cards) >= 17:
            return
        self.dealer_cards.append(self._get_cards(1))

    def _get_observation(self) -> dict:
        """Calculates the current observation based on the player's cards and the dealer's visible card. The observation includes the player's points, the number of not used aces, and the dealer's visible card.
        Returns:
            dict: A dictionary containing the player's points, the number of not used aces, and the dealer's visible card.
        """
        player_usable_aces = self.player_cards.count(11)
        points = sum(self.player_cards)
        if points > MAX_POINTS:
            if player_usable_aces > 0:
                points -= 10
                player_usable_aces -= 1
                self.player_cards[self.player_cards.index(11)] = 1
        return {"player_points": points, "not_used_ace": player_usable_aces, "dealer_card": self.dealer_cards[0]}

    def _get_action_mask(self) -> np.ndarray:
        """Calculates the action mask based on the current state of the environment. The action mask indicates which actions are valid for the current state.
        Returns:
            np.ndarray: An array of shape (5,) where each element is 1 if the corresponding action is valid and 0 otherwise.
        """
        mask = np.ones(5, dtype=np.int8)
        return mask
