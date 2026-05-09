# import gymnasium as gym

# env = gym.make("Blackjack-v1", render_mode="human")

# obs, info = env.reset()

# done = False
# while not done:
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated


import gymnasium as gym
import blackjack_env
import time

env = gym.make("blackjack_env/Blackjack4game-v0", render_mode="human")
obs, info = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1)  # 👈 spowalnia

    if terminated or truncated:
        obs, info = env.reset()

env.close()
