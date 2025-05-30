import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from utils import evaluate_agent

def train_dqn(episodes=500, solve_threshold=195, window=100):
    env = gym.make("CartPole-v1", render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    target_update_freq = 5

    episode_rewards = []
    total_env_steps = 0
    solved_at_step = None

    # For sample efficiency plot
    sample_eff_steps = []
    sample_eff_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps_this_episode = 0

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward
            total_env_steps += 1
            steps_this_episode += 1

        agent.decay_epsilon()

        if episode % target_update_freq == 0:
            agent.update_target_network()

        episode_rewards.append(total_reward)

        # Track sample efficiency
        if len(episode_rewards) >= window:
            avg_reward = np.mean(episode_rewards[-window:])
            sample_eff_steps.append(total_env_steps)
            sample_eff_rewards.append(avg_reward)

            if solved_at_step is None and avg_reward >= solve_threshold:
                solved_at_step = total_env_steps
                print(f"✅ Environment solved at episode {episode}, step {solved_at_step}, avg reward = {avg_reward:.2f}")

        print(f"Episode {episode}: reward = {total_reward:.2f}, epsilon = {agent.epsilon:.3f}, steps = {steps_this_episode}")

    env.close()

    if solved_at_step is None:
        print("❌ Environment not solved during training.")
    else:
        print(f"✅ Sample efficiency: solved in {solved_at_step} environment steps.")

    return episode_rewards, sample_eff_steps, sample_eff_rewards, agent

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Performance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sample_efficiency(steps, rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards)
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Reward (Windowed)")
    plt.title("DQN Sample Efficiency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rewards, steps, avg_rewards, trained_agent = train_dqn()
    plot_rewards(rewards)
    plot_sample_efficiency(steps, avg_rewards)
    evaluate_agent(trained_agent, episodes=5, render=False)
