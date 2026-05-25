import os
import time
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class EvalLoggerCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, eval_freq: int, n_eval_episodes: int, deterministic: bool = True, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = deterministic

        self.timesteps = []
        self.mean_rewards = []
        self.std_rewards = []
        self._last_eval_time = 0
    
    def _on_training_start(self) -> None:
        self._last_eval_time = 0
    
    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_time) >= self.eval_freq:
            self._last_eval_time = self.num_timesteps

            rewards, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=False,
            )
            rewards = np.asarray(rewards, dtype=np.float32)

            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(rewards.mean())
            self.std_rewards.append(rewards.std(ddof=0))
        
        return True

@dataclass
class EvalResults:
    seed: int
    timesteps: np.ndarray
    mean_rewards: np.ndarray
    std_rewards: np.ndarray
    run_dir: str


def test_env_PPO(
    env_train: gym.Env,
    env_eval: gym.Env,
    policy: str,
    verbose: int = 1,
    tensorboard_log: str | None = None,
    n_runs: int = 10,
    run_timesteps: int = 50_000,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 200,
    base_seed: int = 123,
    **ppo_kwargs: dict) -> None:
    
    root_dir = os.path.abspath(f"../results/sb3_blackjack_multi_{int(time.time())}")
    os.makedirs(root_dir, exist_ok=True)
    
    results: list[EvalResults] = []

    for i in range(n_runs):
        print(f"Starting run {i + 1}/{n_runs} with seed {base_seed + i}...")
        seed = base_seed + i
        run_dir = os.path.join(root_dir, f"seed_{seed}_run_{i}")
        os.makedirs(run_dir, exist_ok=True)

        try:
            env_train.reset(seed=seed)
        except TypeError:
            env_train.reset()

        try:
            env_eval.reset(seed=seed + 10_000)
        except TypeError:
            env_eval.reset()

        eval_callback = EvalLoggerCallback(
            eval_env=env_eval,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=verbose,
        )

        model = PPO(
            policy, env_train, verbose=verbose, tensorboard_log=tensorboard_log, seed=seed, **ppo_kwargs
        )
        model.learn(total_timesteps=run_timesteps, callback=eval_callback)

        results.append(
            EvalResults(
                seed=seed,
                timesteps=np.array(eval_callback.timesteps, dtype=np.int32),
                mean_rewards=np.array(eval_callback.mean_rewards, dtype=np.float32),
                std_rewards=np.array(eval_callback.std_rewards, dtype=np.float32),
                run_dir=run_dir,
            )
        )

    env_train.close()
    env_eval.close()
    
    print("Saved runs under:", root_dir)
    print("Eval points per run:", [len(r.timesteps) for r in results])

    plot_eval_results(
        results,
        save_path=os.path.join(root_dir, f"ppo_learning_curve_{time.time()}.png"),
    )


def plot_eval_results(results: list[EvalResults], save_path: str | None = None) -> None:
    T = results[0].timesteps
    Y = np.vstack([r.mean_rewards for r in results])

    mean_across = Y.mean(axis=0)
    std_across = Y.std(axis=0, ddof=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(T, mean_across, linewidth=2, label="mean evaluation reward across runs")
    ax.fill_between(
        T,
        mean_across - std_across,
        mean_across + std_across,
        alpha=0.3,
        label="std deviation across runs",
    )
    ax.set_xlabel("timesteps")
    ax.set_ylabel("mean episode reward")
    ax.set_title("PPO learning curve on Blackjack4game-v1 (mean +- std)")
    ax.grid(True)
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()
    plt.close(fig)
