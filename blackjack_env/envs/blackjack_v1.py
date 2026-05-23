from blackjack_env.envs.blackjack_v0 import BlackjackEnvV0

class BlackjackEnvV1(BlackjackEnvV0):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)