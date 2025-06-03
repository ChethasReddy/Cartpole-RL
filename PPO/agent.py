import torch
import gymnasium as gym
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy

from config import AgentConfig, EvolutionConfig
from network import MlpPolicy
from evolution import EvolutionaryAlgorithm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(AgentConfig, EvolutionConfig):
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.action_size = int(self.env.action_space.n)  # 2 for cartpole

        # Initialize evolutionary algorithm
        self.evolution = EvolutionaryAlgorithm(population_size=self.population_size)
        self.evolution.initialize_population(self.action_size)

        # Initialize PPO components
        self.policy_network = MlpPolicy(action_size=self.action_size).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.k_epoch,
                                                   gamma=0.999)
        self.loss = 0
        self.criterion = nn.MSELoss()
        self.reset_memory()

        self.reward_history = []
        self.avg_reward = []

    def reset_memory(self):
        """Reset the memory buffer"""
        self.memory = {
            'state': [],
            'action': [],
            'reward': [],
            'next_state': [],
            'action_prob': [],
            'terminal': [],
            'advantage': [],
            'td_target': []
        }

    def evaluate_policy(self, policy, num_episodes=5):
        """Evaluate a policy's performance"""
        total_rewards = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(device)
                    action_probs = policy.pi(state_tensor)
                    action = torch.distributions.Categorical(action_probs).sample().item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = bool(terminated or truncated)
                episode_reward += float(reward)
                state = next_state

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)

    def train(self):
        episode = 0
        solved = False

        while not solved:
<<<<<<< HEAD
            print(f"\nEpisode {episode + 1}")
=======
>>>>>>> 3bd563f (made hybrid evolutionary reinforcement Algorithm using PPO)
            # Evolutionary Phase
            for gen in range(self.evolution_epochs):
                # Evaluate all policies in the population
                for i, policy in enumerate(self.evolution.population):
                    fitness = self.evaluate_policy(policy)
                    self.evolution.fitness_scores[i] = float(fitness)

                # Update best policy
                self.evolution.update_best_policy()

                # Evolve population
                self.evolution.evolve()

                print(f'Generation {gen + 1}, Best Fitness: {self.evolution.best_fitness:.2f}')

            # PPO Fine-tuning Phase
            # Select top policies for PPO fine-tuning
            sorted_indices = np.argsort(self.evolution.fitness_scores)[-self.num_ppo_policies:]
            top_policies = [self.evolution.population[i] for i in sorted_indices]

            for policy in top_policies:
                # Copy policy weights to PPO network
                self.policy_network.load_state_dict(policy.state_dict())

                # Fine-tune with PPO
                for _ in range(self.ppo_epochs):
<<<<<<< HEAD
                    reward = self.train_ppo_episode()
                    print(f"PPO Episode Reward: {reward:.2f}")
=======
                    self.train_ppo_episode()
>>>>>>> 3bd563f (made hybrid evolutionary reinforcement Algorithm using PPO)

                # Update policy in population with fine-tuned weights
                policy.load_state_dict(self.policy_network.state_dict())

            # Check if solved
            if self.evolution.best_fitness >= 195:
                solved = True
                print("Environment solved!")

            # Plot progress
            if episode % self.plot_every == 0:
                plot_graph(self.reward_history, self.avg_reward)

            episode += 1

        self.env.close()

    def train_ppo_episode(self):
        """Train one episode using PPO"""
        state, _ = self.env.reset()
        episode_reward = 0
        done = False

        # Reset memory for new episode
        self.reset_memory()

        while not done:
            # Choose action
            prob_a = self.policy_network.pi(torch.FloatTensor(state).to(device))
            action = torch.distributions.Categorical(prob_a).sample().item()

            # Act
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            terminal = bool(terminated or truncated)
            reward = float(reward)

            reward = -1.0 if terminal else reward

            self.add_memory(state, action, reward/10.0, next_state, terminal, prob_a[action].item())

            state = next_state
            episode_reward += reward

            if terminal:
                self.finish_path()
                self.update_network()
                break

        self.reward_history.append(episode_reward)
        self.avg_reward.append(sum(self.reward_history[-10:])/10.0)

        return episode_reward

    def add_memory(self, s, a, r, next_s, t, prob):
        """Add a transition to memory"""
        self.memory['state'].append(s)
        self.memory['action'].append([a])
        self.memory['reward'].append([r])
        self.memory['next_state'].append(next_s)
        self.memory['terminal'].append([1 - t])
        self.memory['action_prob'].append([prob])

    def finish_path(self):
        """Calculate advantages and TD targets for the episode"""
        states = torch.FloatTensor(self.memory['state']).to(device)
        rewards = torch.FloatTensor(self.memory['reward'])
        next_states = torch.FloatTensor(self.memory['next_state']).to(device)
        terminals = torch.FloatTensor(self.memory['terminal'])

        # Calculate TD targets
        with torch.no_grad():
            next_values = self.policy_network.v(next_states).squeeze()
            values = self.policy_network.v(states).squeeze()

        td_target = rewards + self.gamma * next_values * terminals
        advantages = td_target - values

        self.memory['td_target'] = td_target.tolist()
        self.memory['advantage'] = advantages.tolist()

    def update_network(self):
        """Update the network using PPO"""
        # Convert memory to tensors
        states = torch.FloatTensor(self.memory['state']).to(device)
        actions = torch.LongTensor(self.memory['action']).to(device)
        old_probs = torch.FloatTensor(self.memory['action_prob']).to(device)
        advantages = torch.FloatTensor(self.memory['advantage']).to(device)
        td_targets = torch.FloatTensor(self.memory['td_target']).to(device)

        # Get new action probabilities
        new_probs = self.policy_network.pi(states)
        new_probs_a = torch.gather(new_probs, 1, actions)

        # Calculate ratio
        ratio = torch.exp(torch.log(new_probs_a) - torch.log(old_probs))

        # Calculate surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        # Calculate value loss
        pred_v = self.policy_network.v(states)
        v_loss = 0.5 * (pred_v - td_targets).pow(2)

        # Calculate entropy
        entropy = torch.distributions.Categorical(new_probs).entropy()

        # Total loss
        self.loss = (-torch.min(surr1, surr2).mean() +
                    self.v_coef * v_loss.mean() -
                    self.entropy_coef * entropy.mean())

        # Optimize
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()


def plot_graph(reward_history, avg_reward):
    df = pd.DataFrame({'x': range(len(reward_history)), 'Reward': reward_history, 'Average': avg_reward})
    plt.style.use('default')  # Using default matplotlib style

    plt.figure(figsize=(10, 6))
    plt.plot(df['x'], df['Reward'], marker='', color='blue', linewidth=0.8, alpha=0.9, label='Reward')
    plt.plot(df['x'], df['Average'], marker='', color='red', linewidth=1, alpha=0.9, label='Average')

    plt.legend(loc='upper left')
    plt.title("CartPole Training Progress", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.savefig('score.png')
    plt.close()
