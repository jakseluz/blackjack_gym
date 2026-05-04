from gymnasium.envs.registration import register

register(
    id="blackjack_env/GridWorld-v0",
    entry_point="blackjack_env.envs:GridWorldEnv",
)
