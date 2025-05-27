# train_dqn.py

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from cartpole_env import create_cartpole_env  # optional helper


def train_dqn(episodes=500):
    env = create_cartpole_env(render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    target_update_freq = 5

    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        if episode % target_update_freq == 0:
            agent.update_target_network()

        episode_rewards.append(total_reward)
        print(f"Episode {episode}: reward = {total_reward:.2f}, epsilon = {agent.epsilon:.3f}")

    env.close()
    return episode_rewards, agent


def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Performance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_agent(agent, episodes=5):
    env = create_cartpole_env(render_mode="human")
    agent.epsilon = 0.0  # disable exploration

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Evaluation Episode {ep}: Total Reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    rewards, trained_agent = train_dqn()
    plot_rewards(rewards)
    evaluate_agent(trained_agent)
