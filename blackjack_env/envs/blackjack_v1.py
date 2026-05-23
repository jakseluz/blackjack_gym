from gymnasium import spaces
import numpy as np

from blackjack_env.envs.blackjack_v0 import BlackjackEnvV0

class BlackjackEnvV1(BlackjackEnvV0):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 4.99]),
            dtype=np.float32,
        )
        self.phase = "betting"

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        bet_percentage = action[0]
        game_decision = int(action[1])

        if self.phase == "betting":
            self.phase = "playing"
            self.bet_percentage = bet_percentage
            return self._return_step_info(reward=0.0, terminated=False, action=action)
        else:
            return super().step(
                game_decision,
                reward_percentage=self.bet_percentage,
            )
