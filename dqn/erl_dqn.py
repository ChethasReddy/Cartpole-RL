import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import numpy as np
from collections import deque
from models import DQN

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.fitness = -np.inf  # Used in evolution

    def act(self, state, exploit=False):
        if not exploit and np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return self.q_network(state_tensor).argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q = self.q_network(states).gather(1, actions)
        next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def clone(self):
        clone = DQNAgent(self.state_dim, self.action_dim)
        clone.q_network.load_state_dict(self.q_network.state_dict())
        return clone

class EvolutionManager:
    def __init__(self, env, state_dim, action_dim, agent_class, population_size=10, elite_frac=0.2, mutation_rate=0.1):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_class = agent_class
        self.population_size = population_size
        self.elite_size = int(elite_frac * population_size)
        self.mutation_rate = mutation_rate
        self.population = [agent_class(state_dim, action_dim) for _ in range(population_size)]

    def evaluate_fitness(self, agent, train_episodes=5, eval_episodes=3):
        steps_used = 0
        agent.epsilon = 1.0
        for _ in range(train_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                agent.remember(state, action, reward, next_state, done)
                agent.update()
                state = next_state
                steps_used += 1
            agent.decay_epsilon()
        agent.update_target_network()

        # Evaluation
        total_reward = 0
        agent.epsilon = 0.0
        for _ in range(eval_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = agent.act(state, exploit=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps_used += 1

        avg_reward = total_reward / eval_episodes
        agent.fitness = avg_reward
        return avg_reward, steps_used

    def evolve(self):
        total_steps = 0
        for agent in self.population:
            _, steps = self.evaluate_fitness(agent)
            total_steps += steps

        self.population.sort(key=lambda agent: agent.fitness, reverse=True)
        elites = self.population[:self.elite_size]

        new_population = elites.copy()
        while len(new_population) < self.population_size:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)

        self.population = new_population
        return total_steps

    def mutate(self, agent):
        for param in agent.q_network.parameters():
            noise = torch.randn_like(param) * self.mutation_rate
            param.data += noise

    def crossover(self, parent1, parent2):
        """Create a new agent by mixing parameters from two parents."""
        child = parent1.clone()
        with torch.no_grad():
            for child_param, p1_param, p2_param in zip(child.q_network.parameters(),
                                                       parent1.q_network.parameters(),
                                                       parent2.q_network.parameters()):
                mask = torch.rand_like(child_param) < 0.5
                child_param.copy_(torch.where(mask, p1_param, p2_param))
        return child

    def get_best_agent(self):
        return max(self.population, key=lambda agent: agent.fitness)
