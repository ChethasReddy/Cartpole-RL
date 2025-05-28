import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Define the neural network for DQN
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.model(x)

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

    def clone(self):
        """Clone the agent, including network parameters."""
        clone = DQNAgent(self.state_dim, self.action_dim)
        clone.q_network.load_state_dict(self.q_network.state_dict())
        return clone
