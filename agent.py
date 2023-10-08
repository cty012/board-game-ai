import random
import torch
import torch.nn as nn

from game import normalize


class QLearningAgent:
    def __init__(self, network, device, lr, gamma):
        self.network = network
        self.device = device
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma

    def sample(self, game, player, epsilon):
        """Sample from the game."""
        valid_actions = game.get_valid_actions(player)
        if len(valid_actions) == 0:
            return 0

        # Exploration
        if torch.rand(1).item() < epsilon:
            rand_index = torch.randint(0, len(valid_actions), (1,)).item()
            return valid_actions[rand_index]

        # Exploitation
        else:
            # Set the network in evaluation mode
            self.network.eval()

            # Disable gradient for higher efficiency
            with torch.no_grad():
                state_tensor = game.get_state().unsqueeze(0).to(self.device)  # Add batch dimension
                action = self.network.sample(state_tensor, valid_actions)

            # Greedy action based on Q-values
            return action

    def train(self, batch):
        """Train the network using the batch of samples."""
        # Set the network in training mode
        self.network.train()

        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors and compute Q-values
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.stack([normalize(state, 1) for state in next_states]).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        q_values = self.network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        next_q_values = self.network(next_states).max(dim=1)[0]

        # Compute target Q-values
        # Why -1? Because in zero-sum games, the opponent's gain = the player's loss
        targets = rewards + self.gamma * (1 - dones) * (-1) * next_q_values

        # Compute loss
        loss = nn.functional.mse_loss(q_values, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, experience):
        """Pushes a new experience for future sampling."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def sample(self, batch_size):
        """Generate a batch of samples from the current experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))


class Epsilon:
    def __init__(self, epsilon_max, epsilon_min, epsilon_decay):
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_max

    def reset(self):
        """Reset to initial value"""
        self.epsilon = self.epsilon_max

    def update(self):
        """Update the epsilon after an episode"""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def get(self):
        """Get the epsilon value"""
        return self.epsilon
