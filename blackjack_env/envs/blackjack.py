import gymnasium as gym
from gymnasium import spaces
import numpy as np

from blackjack_env.envs.settings import (
    MAX_POINTS,
    DEALER_LIMIT,
    Actions,
    Game_version,
    Casino_ace,
    Reward,
    SurrenderRule,
)


class BlackjackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        game_version=Game_version.AMERICAN,
        casino_ace=Casino_ace.STAND_ON_ALL_17S,
        blackjack_reward=Reward.blackjack32,
        surrender_rule=SurrenderRule.NOT_AVAILABLE,
    ):
        """Initializes the Blackjack environment. The observation space consists of the player's points, the number of not used aces, and the dealer's visible card. The action space consists of 5 actions: hit, stand, double down, split, and insurance.
        Args:
            render_mode (str, optional): The mode to render the environment. Defaults to None.
            game_version (Game_version, optional): The version of the game to play. Defaults to Game_version.AMERICAN.
            casino_ace (Casino_ace, optional): The rule for the dealer's behavior with aces. Defaults to Casino_ace.STAND_ON_ALL_17S.
            blackjack_reward (Reward, optional): The reward for getting a blackjack. Defaults to Reward.blackjack32.
            surrender_rule (SurrenderRule, optional): The rule for surrendering. Defaults to SurrenderRule.NOT_AVAILABLE.
        """
        self.observation_space = spaces.Dict(
            {
                "player_points": spaces.Discrete(31),
                "not_used_ace": spaces.Discrete(4),
                "dealer_card": spaces.Discrete(11, start=1),
            }
        )

        self.action_space = spaces.Discrete(5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        assert game_version in Game_version
        self.game_version = game_version

        assert casino_ace in Casino_ace
        self.casino_ace = casino_ace

        assert blackjack_reward in Reward
        self.blackjack_reward = blackjack_reward

        assert surrender_rule in SurrenderRule
        self.surrender_rule = surrender_rule

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
        picked = self._get_cards(3) if self.game_version == Game_version.AMERICAN else self._get_cards(4)
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
        terminated = False
        reward = 0.0

        if self.game_version == Game_version.AMERICAN:
            if self._is_blackjack(self.dealer_cards):
                reward += self._dealer_moves()
                return self._return_step_info(reward=reward, terminated=True)

        if action == Actions.HIT.value:
            self.player_cards.append(self._get_cards(1))
        elif action == Actions.STAND.value:
            reward += self._dealer_moves()
            terminated = True
        elif action == Actions.DOUBLE_DOWN.value:
            self.player_cards.append(self._get_cards(1))
            reward += self._dealer_moves()
            terminated = True
            reward *= 2
        elif action == Actions.SPLIT.value:
            # TODO
            pass
        elif action == Actions.INSURANCE.value:
            if not self._is_blackjack(self.dealer_cards):
                reward = -0.5
            else:
                terminated = True

        return self._return_step_info(reward=reward, terminated=terminated)

    def _return_step_info(self, reward: float, terminated: bool) -> tuple[dict, float, bool, bool, dict]:
        """Helper method to return the step information in the correct format.
        Args:
            reward (float): The reward to return.
            terminated (bool): Whether the episode has terminated.
        Returns:
            tuple[dict, float, bool, bool, dict]: A tuple containing the resulting observation, reward, terminated, truncated, and info.
        """
        observation = self._get_observation()
        info = {"action_mask": self._get_action_mask()}
        truncated = False
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
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

    def _dealer_moves(self) -> float:
        """Performs the dealer's move according to the rules of Blackjack and calculates the outcome. The dealer will keep drawing cards until their points are {DEALER_LIMIT} or higher. Then the game ends.
        You probably want to add a result of the method to a variable, not to replace its value.
        Returns:
            float: a reward value gained at the end of a game
        """
        while sum(self.dealer_cards) < DEALER_LIMIT:
            self.dealer_cards.append(self._get_cards(1))

        reward = 0.0
        player_score = sum(self.player_cards)
        dealer_score = sum(self.player_cards)
        if player_score > MAX_POINTS:
            reward = -1.0
        elif dealer_score > MAX_POINTS:
            reward = 1.0
        elif player_score > dealer_score:
            if self._is_blackjack(self.player_cards):
                reward = self.blackjack_reward
            else:
                reward = 1.0
        elif player_score == dealer_score:
            reward = 0.5  # experimental value
        else:
            reward = -1.0

        return reward

    def _is_blackjack(self, cards: list[int]) -> bool:
        """Checks if there is a blackjack situation in a given deck
        Args:
            cards (list[int]): a given deck
        Returns:
            bool: True if there is a blackjack, False otherwise
        """
        if set(cards) in ({1, 10}, {11, 10}):
            return True
        return False

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
        if sum(self.player_cards) >= MAX_POINTS:
            mask[Actions.HIT.value] = 0
        if len(self.player_cards) > 2:
            mask[Actions.DOUBLE_DOWN.value] = 0
            mask[Actions.SPLIT.value] = 0
            mask[Actions.INSURANCE.value] = 0
        if self.dealer_cards[0] not in (1, 11):
            mask[Actions.INSURANCE.value] = 0
        if self.player_cards[0] != self.player_cards[1]:
            mask[Actions.SPLIT.value] = 0

        mask[Actions.SPLIT.value] = 0  # TODO

        return mask
