import gymnasium as gym
from gymnasium import spaces
import numpy as np

from blackjack_env.envs.settings import MAX_POINTS

class ShiftWrapper(gym.Wrapper):
    """Allow to use Discrete() observation spaces with start!=0"""
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "player_points": spaces.Discrete(MAX_POINTS + 2),
                "not_used_ace": spaces.Discrete(4),
                "dealer_card": spaces.Discrete(11),
            }
        )

        if self.env.unwrapped.intelligence_mode:
            self.observation_space = spaces.Dict(
                {
                    "player_points": spaces.Discrete(MAX_POINTS + 2),
                    "not_used_ace": spaces.Discrete(4),
                    "dealer_card": spaces.Discrete(11),
                    "running_count": spaces.Discrete(41),
                }
            )
        
    def _shift_observation(self, observation: tuple[dict, float, bool, bool, dict]) -> tuple[dict, float, bool, bool, dict]:
        """Shift the observation to match the new observation space."""
        shifted_observation = {
            "player_points": observation["player_points"],
            "not_used_ace": observation["not_used_ace"],
            "dealer_card": observation["dealer_card"] - 1,
        }

        if self.env.unwrapped.intelligence_mode:
            shifted_observation["running_count"] = observation["running_count"] + 20

        return shifted_observation

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._shift_observation(observation), info
    
    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """Shift the observation returned by the environment."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._shift_observation(observation), reward, terminated, truncated, info
    
    def action_masks(self) -> np.ndarray:
        """
        MaskablePPO (sb3-contrib) uses this method to get valid actions.
        Must return a 1D array of bools/0-1 with shape (n_actions,).
        """
        mask = self._get_action_mask()
        return mask.astype(bool)
        