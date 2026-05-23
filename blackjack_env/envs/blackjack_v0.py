import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
import time

from blackjack_env.envs.settings import (
    MAX_POINTS,
    DEALER_LIMIT,
    Actions,
    Game_version,
    Casino_ace,
    Reward,
    SurrenderRule,
)


class BlackjackEnvV0(gym.Env):
    metadata = {"render_modes": ["human", "terminal"], "render_fps": 10}

    def __init__(
        self,
        render_mode=None,
        game_version=Game_version.AMERICAN,
        casino_ace=Casino_ace.STAND_ON_ALL_17S,
        blackjack_reward=Reward.blackjack32,
        surrender_rule=SurrenderRule.NOT_AVAILABLE,
        intelligence_mode=True,
    ):
        """Initializes the Blackjack environment. The observation space consists of the player's points, the number of not used aces, and the dealer's visible card. The action space consists of 5 actions: hit, stand, double down, split, and insurance.
        Args:
            render_mode (str, optional): The mode to render the environment. Defaults to None.
            game_version (Game_version, optional): The version of the game to play. Defaults to Game_version.AMERICAN.
            casino_ace (Casino_ace, optional): The rule for the dealer's behavior with aces. Defaults to Casino_ace.STAND_ON_ALL_17S.
            blackjack_reward (Reward, optional): The reward for getting a blackjack. Defaults to Reward.blackjack32.
            surrender_rule (SurrenderRule, optional): The rule for surrendering. Defaults to SurrenderRule.NOT_AVAILABLE.
            intelligence_mode (bool, optional): Whether to include the running count in the observation space. Defaults to False.
        """
        self.observation_space = spaces.Dict(
            {
                "player_points": spaces.Discrete(MAX_POINTS + 2),
                "not_used_ace": spaces.Discrete(4),
                "dealer_card": spaces.Discrete(11, start=1),
            }
        )

        self.action_space = spaces.Discrete(5)

        self.intelligence_mode = intelligence_mode

        if intelligence_mode:
            # Karty 2, 3, 4, 5, 6 mają wartość +1
            # Karty 7, 8, 9 mają wartość 0
            # Karty 10, Walet, Dama, Król, As mają wartość -1
            self.card_costs = {
                2: 1,
                3: 1,
                4: 1,
                5: 1,
                6: 1,
                7: 0,
                8: 0,
                9: 0,
                10: -1,
                11: -1,
                1: -1,
            }
            self.observation_space = spaces.Dict(
                {
                    "player_points": spaces.Discrete(MAX_POINTS + 2),
                    "not_used_ace": spaces.Discrete(4),
                    "dealer_card": spaces.Discrete(11, start=1),
                    "running_count": spaces.Discrete(41, start=-20),
                }
            )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            # Pygame setup variables
            self.window_size = 512  # Size of the Pygame window
            self.window = pygame.display.set_mode((self.window_size, self.window_size))  # The window object
            self.clock = pygame.time.Clock()  # Controls the framerate
            image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "playingcards_bridgesize_png"))
            self.card_images = {
                l
                + "-"
                + str(num): pygame.transform.scale(
                    pygame.image.load(os.path.join(image_dir, l + "-" + str(num) + ".png")), (90, 150)
                )
                for num in range(1, 14)
                for l in ("C", "D", "H", "S")
            }
            self.card_back = pygame.transform.scale(pygame.image.load(os.path.join(image_dir, "Back-B.png")), (90, 150))
            self.drawn_cards = {
                "player": [],
                "dealer": [],
            }

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
        self.status = "playing"
        if self.intelligence_mode:
            self.running_count = 0
        if self.render_mode == "human":
            self.drawn_cards = {
                "player": [],
                "dealer": [],
            }

        self.deck = {
            (l + "-" + str(num)): (num if num <= 10 else 10) for num in range(1, 14) for l in ("C", "D", "H", "S")
        }
        picked = self._get_cards([2, 2]) if self.game_version == Game_version.AMERICAN else self._get_cards([2, 1])

        self.player_cards = picked[:2]
        self.dealer_cards = picked[2:]

        observation = self._get_observation()
        info = self._get_info()
        self._refresh_human_render()
        return observation, info

    def step(
        self, action, reward_percentage=1.0
    ) -> tuple[dict, float, bool, bool, dict]:
        """Performs the given action in the environment and returns the resulting observation, reward, terminated, truncated, and info.
        Args:
            action (int): The action to perform, represented as an integer corresponding to the Actions enum.
        Returns:
            tuple[dict, float, bool, bool, dict]: A tuple containing the resulting observation, reward, terminated, truncated, and info.
        """
        mask = self._get_action_mask()
        if mask[action] == 0:
            raise ValueError(f"Illegal action {action} for this state (masked out).")

        terminated = False
        reward = 0.0

        if self.game_version == Game_version.AMERICAN and len(self.player_cards) == 2 and len(self.dealer_cards) == 2:
            player_bj = self._is_blackjack(self.player_cards)
            dealer_bj = self._is_blackjack(self.dealer_cards)

            if player_bj:
                reward = (
                    0.0
                    if dealer_bj
                    else float(self.blackjack_reward.value) * reward_percentage
                )
                return self._return_step_info(reward=reward, terminated=True, action=action)

            # dealer peek (upraszczamy): jeśli dealer ma BJ, to tylko INSURANCE może zmienić wynik
            if dealer_bj:
                if action == Actions.INSURANCE.value and self.dealer_cards[0] == 11:
                    reward = 0.0  # net 0: -1 (bet) +1 (insurance)
                else:
                    reward = -1.0 * reward_percentage
                return self._return_step_info(reward=reward, terminated=True, action=action)

        if action == Actions.HIT.value:
            self.player_cards.append(self._get_cards([1, 0]))
            self._refresh_human_render()
        elif action == Actions.STAND.value:
            self._refresh_human_render()
            reward += self._dealer_moves() * reward_percentage
            terminated = True
        elif action == Actions.DOUBLE_DOWN.value:
            self.player_cards.append(self._get_cards([1, 0]))
            self._refresh_human_render()
            reward += self._dealer_moves() * reward_percentage
            terminated = True
            reward *= 2
        elif action == Actions.SPLIT.value:
            # TODO
            pass
        elif action == Actions.INSURANCE.value:
            # jeśli doszliśmy tutaj, to dealer nie miał BJ (w AMERICAN), więc insurance przegrywa
            reward = -0.5 * reward_percentage
            self._refresh_human_render()

        return self._return_step_info(reward=reward, terminated=terminated, action=action)

    def _return_step_info(self, reward: float, terminated: bool, action: int) -> tuple[dict, float, bool, bool, dict]:
        """Helper method to return the step information in the correct format.
        Args:
            reward (float): The reward to return.
            terminated (bool): Whether the episode has terminated.
            action (int): The action that was taken.
        Returns:
            tuple[dict, float, bool, bool, dict]: A tuple containing the resulting observation, reward, terminated, truncated, and info.
        """
        observation = self._get_observation()
        info = self._get_info(action=action)
        truncated = False
        return observation, reward, terminated, truncated, info

    def _get_info(self, action: int | None = None) -> dict:
        """Helper method to return info for the outer world. The info includes the action mask, the number of cards left in the deck, the player's cards, and the dealer's cards.
        Args:
            action (int | None, optional): The action that was taken. Defaults to None.
        Returns:
            dict: A dictionary containing the action mask, the number of cards left in the deck, the player's cards, and the dealer's cards.
        """
        return {
            "action_mask": self._get_action_mask(),
            "deck_status": len(self.deck),
            "player_cards": self.player_cards,
            "dealer_cards": self.dealer_cards,
            "taken_action": action,
        }

    def render(self) -> None:
        """Renders the current state of the environment."""
        if self.render_mode == "terminal":
            print(f"Player's cards: {self.player_cards},\nDealer's visible card: {self.dealer_cards[0]}\n")

    def _refresh_human_render(self, time_interval: float = 3.0) -> None:
        """Refreshes the Pygame rendering of the environment. This method is called after every action to update the visual representation of the game state.
        Args:
            time_interval (float, optional): The amount of time to wait after rendering the frame. Defaults to 3.0 seconds.
        """
        if self.render_mode == "human":
            self._render_frame()
            time.sleep(time_interval)

    def _get_cards(self, num: list[int]) -> int | list[int]:
        """Draws a specified number of cards from the deck and returns them. The drawn cards are removed from the deck.
        Args:
            num (list[int]): A list of integers specifying the number of cards to draw for each player. The length of the list should be equal to the number of players (2 by default).
        Returns:
            int | list[int]: A single card if num is 1, otherwise a list of drawn cards.
        """
        idxs = self.np_random.choice(list(self.deck.keys()), size=sum(num), replace=False)
        picked = [self.deck[i] for i in idxs]
        picked = [11 if c == 1 else c for c in picked]

        if self.render_mode == "human":
            images = [self.card_images[i] for i in idxs]
            j = 0
            for i, img in enumerate(images):
                if i >= sum(num[: j + 1]):
                    j += 1
                self.drawn_cards[list(self.drawn_cards.keys())[j]].append(img)

        for v in idxs:
            del self.deck[v]
            if self.render_mode == "human":
                del self.card_images[v]

        if self.intelligence_mode:
            for i, card in enumerate(picked):
                if self.game_version == Game_version.AMERICAN and num == [2, 2] and i >= 3:
                    break
                self.running_count += self.card_costs[card]

        return picked[0] if len(picked) == 1 else picked

    def _dealer_moves(self) -> float:
        """Performs the dealer's move according to the rules of Blackjack and calculates the outcome. The dealer will keep drawing cards until their points are {DEALER_LIMIT} or higher. Then the game ends.
        You probably want to add a result of the method to a variable, not to replace its value.
        Returns:
            float: a reward value gained at the end of a game
        """
        self.status = "dealer_turn"
        if self.intelligence_mode and self.game_version == Game_version.AMERICAN:
            self.running_count += self.card_costs[self.dealer_cards[1]]
        self._refresh_human_render()

        dealer_points, _ = self._handle_value_and_usable_aces(self.dealer_cards)
        while dealer_points < DEALER_LIMIT:
            self.dealer_cards.append(self._get_cards([0, 1]))
            dealer_points, _ = self._handle_value_and_usable_aces(self.dealer_cards)
            self._refresh_human_render()

        reward = 0.0
        player_score, _ = self._handle_value_and_usable_aces(self.player_cards)
        dealer_score, _ = self._handle_value_and_usable_aces(self.dealer_cards)
        if player_score > MAX_POINTS:
            reward = -1.0
        elif dealer_score > MAX_POINTS:
            reward = 1.0
        elif player_score > dealer_score:
            reward = self.blackjack_reward.value if self._is_blackjack(self.player_cards) else 1.0
        elif player_score == dealer_score:
            reward = 0.0
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
        return len(cards) == 2 and set(cards) == {10, 11}

    def _get_observation(self) -> dict:
        """Calculates the current observation based on the player's cards and the dealer's visible card. The observation includes the player's points, the number of not used aces, and the dealer's visible card.
        Returns:
            dict: A dictionary containing the player's points, the number of not used aces, and the dealer's visible card.
        """
        points, player_usable_aces = self._handle_value_and_usable_aces(self.player_cards)

        if points > MAX_POINTS:
            points = MAX_POINTS + 1

        observation = {
            "player_points": points,
            "not_used_ace": player_usable_aces,
            "dealer_card": self.dealer_cards[0],
        }
        if self.intelligence_mode:
            observation["running_count"] = self.running_count
        return observation

    def _handle_value_and_usable_aces(self, cards: list[int]) -> tuple[int, int]:
        """Calculates the total points of a hand and the number of usable aces. A usable ace is an ace that can be counted as 11 without the hand going over 21.
        Args:
            cards (list[int]): A list of integers representing the cards in the hand.
        Returns:
            tuple[int, int]: A tuple containing the total points and the number of usable aces.
        """
        points = sum(cards)
        usable_aces = cards.count(11)
        while points > MAX_POINTS and usable_aces > 0:
            points -= 10
            usable_aces -= 1
        return points, usable_aces

    def _get_action_mask(self) -> np.ndarray:
        """Calculates the action mask based on the current state of the environment. The action mask indicates which actions are valid for the current state.
        Returns:
            np.ndarray: An array of shape (5,) where each element is 1 if the corresponding action is valid and 0 otherwise.
        """
        mask = np.ones(5, dtype=np.int8)
        player_points, _ = self._handle_value_and_usable_aces(self.player_cards)

        if player_points >= MAX_POINTS:
            mask[Actions.HIT.value] = 0
            mask[Actions.DOUBLE_DOWN.value] = 0
            mask[Actions.SPLIT.value] = 0
            mask[Actions.INSURANCE.value] = 0
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

    def _render_frame(self) -> None:
        """Renders the current state of the environment using Pygame."""
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 255, 0))

        font = pygame.font.Font(None, 36)
        text_title = font.render("Blackjack", True, (0, 0, 0))
        canvas.blit(text_title, (200, 20))

        player_text = font.render("Player", True, (0, 0, 0))
        canvas.blit(player_text, (50, 260))
        for i, card in enumerate(self.drawn_cards["player"]):
            canvas.blit(card, (50 + i * 60, 300))

        dealer_text = font.render("Dealer", True, (0, 0, 0))
        canvas.blit(dealer_text, (50, 60))
        for i, card in enumerate(self.drawn_cards["dealer"]):
            if i > 0 and self.status == "playing":
                canvas.blit(self.card_back, (50 + i * 60, 100))
                break
            canvas.blit(card, (50 + i * 60, 100))

        pygame.event.pump()
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human":
            if self.window is not None:
                pygame.font.quit()
                pygame.display.quit()
                pygame.quit()
