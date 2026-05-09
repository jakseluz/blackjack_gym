from gymnasium.envs.registration import register

register(
    id="blackjack_env/Blackjack4game-v0",
    entry_point="blackjack_env.envs:BlackjackEnv",
)
