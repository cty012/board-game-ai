import random
import torch
import torch.nn as nn


class QLearningAgent:
    def __init__(self, network, lr, gamma):
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma

    def sample(self, game, player, epsilon):
        """Sample from the game."""
        # Exploration
        if torch.rand(1).item() < epsilon:
            valid_moves = game.get_valid_moves(player)
            if len(valid_moves) == 0:
                return 0
            return valid_moves[torch.randint(0, len(valid_moves), (1,)).item()]

        # Exploitation
        else:
            # Set the network in evaluation mode
            self.network.eval()

            # Disable gradient for higher efficiency
            with torch.no_grad():
                state_tensor = game.get_state().unsqueeze(0)  # Add batch dimension
                q_values = self.network(state_tensor)

            # Greedy action based on Q-values
            return torch.argmax(q_values).item()

    def train(self, batch):
        """Train the network using the batch of samples."""
        # Set the network in training mode
        self.network.train()

        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors and compute Q-values
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float)

        _q_val = self.network(states)
        q_values = _q_val.gather(1, actions.unsqueeze(-1)).squeeze(-1)

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
