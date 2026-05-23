from gymnasium.envs.registration import register

register(
    id="Blackjack4game-v0",
    entry_point="blackjack_env.envs:BlackjackEnvV0",
)

register(
    id="Blackjack4game-v1",
    entry_point="blackjack_env.envs:BlackjackEnvV1",
)