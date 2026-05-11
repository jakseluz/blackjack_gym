import numpy as np
import gymnasium as gym
import time
import os
import matplotlib.pyplot as plt


def q_learning(
    env: gym.Env, num_episodes: int, epsilon=1.0, alpha=0.1, gamma=0.9, epsilon_decay=0.999
) -> tuple[np.ndarray, list[float]]:
    """Trains a Q-learning agent on the given environment.
    Args:
        env (gym.Env): The environment to train on.
        num_episodes (int): The number of episodes to train for.
        epsilon (float, optional): The initial exploration rate. Defaults to 1.0.
        alpha (float, optional): The learning rate. Defaults to 0.1.
        gamma (float, optional): The discount factor. Defaults to 0.9.
        epsilon_decay (float, optional): The decay rate for epsilon after each episode. Defaults to 0.999.
    Returns:
        tuple[np.ndarray, list[float]]: A tuple containing the learned Q-table and a list of rewards obtained in each episode.
    """
    n_player_points = env.observation_space["player_points"].n
    n_not_used_aces = env.observation_space["not_used_ace"].n
    n_dealer_cards = env.observation_space["dealer_card"].n
    n_running_counts = env.observation_space["running_count"].n if "running_count" in env.observation_space else 1
    total_states = n_player_points * n_not_used_aces * n_dealer_cards * n_running_counts
    q_table = np.zeros((total_states, env.action_space.n))

    rewards_history = []

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_index = get_state_index(state)

            action_mask = info["action_mask"].astype(bool)
            allowed_actions = np.flatnonzero(action_mask)

            if env.np_random.random() < epsilon:
                action = env.action_space.sample()
                while action_mask[action] == 0:
                    action = env.action_space.sample()
            else:
                action = allowed_actions[np.argmax(q_table[state_index][allowed_actions])]

            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            next_action_mask = info["action_mask"].astype(bool)
            next_allowed_actions = np.flatnonzero(next_action_mask)

            if done:
                next_max = 0.0
            else:
                next_state_index = get_state_index(next_state)
                next_max = np.max(q_table[next_state_index][next_allowed_actions])
            # except IndexError as e:
            #     print("IndexError:", e)
            #     print("next_state:", next_state)
            #     print("get_state_index(next_state):", get_state_index(next_state))
            #     print("Q-table shape:", q_table.shape)
            #     raise

            old_value = q_table[state_index, action]
            q_table[state_index, action] = old_value + alpha * (reward + gamma * next_max - old_value)

            state = next_state
        rewards_history.append(total_reward)
        epsilon *= epsilon_decay

    os.makedirs("results/q_learning", exist_ok=True)
    np.save("results/q_learning/q_table_{}.npy".format(time.strftime("%Y-%m-%d_%H-%M-%S")), q_table)
    np.save(
        "results/q_learning/rewards_history_{}.npy".format(time.strftime("%Y-%m-%d_%H-%M-%S")),
        np.array(rewards_history),
    )
    return q_table, rewards_history


def get_state_index(state):
    """Converts the state dictionary to a unique index for the Q-table.
    Args:
        state (dict): The state dictionary containing player points, not used aces, dealer card, and optionally running count.
    Returns:
        int: A unique index representing the state.
    """
    p = state["player_points"]
    d = state["not_used_ace"]
    a = state["dealer_card"] - 1
    r = state["running_count"] + 20 if "running_count" in state else 0
    m = 41 if "running_count" in state else 1
    # p * (max_d * max_a * max_r) + d * (max_a * max_r) + a * max_r + r
    return p * (4 * 11 * m) + d * 11 * m + a * m + r


def plot_rewards_history(rewards_history: list[float]) -> None:
    """Plots the rewards history.
    Args:
        rewards_history (list[float]): A list of rewards obtained in each episode.
    """
    plt.figure(figsize=(50, 6))
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Rewards History Q-learning agent")
    plt.grid()
    plt.show()
