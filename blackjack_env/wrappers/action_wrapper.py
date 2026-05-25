import numpy as np
import gymnasium as gym

class SafeV1ActionWrapper(gym.Wrapper):
    """
    Works with Blackjack4game-v1 (Box(2,)).
    - clips bet to [0, 1]
    - forces decision into {0..4}
    - if decision is illegal in current state, replaces it with first legal action
    """
    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        bet = float(np.clip(a[0], 0.0, 1.0))
        decision = int(np.clip(np.floor(float(a[1])), 0, 4))

        phase = getattr(self.env.unwrapped, "phase", "playing")

        if phase == "betting":
            # decision ignored by env in betting phase; keep it stable
            decision = 0
        else:
            mask = self.env.unwrapped._get_action_mask()  # shape (5,)
            if mask[decision] == 0:
                decision = int(np.flatnonzero(mask)[0])

        fixed_action = np.array([bet, float(decision)], dtype=np.float32)
        return self.env.step(fixed_action)