import random
import torch
import torch.nn as nn


class QLearningAgent:
    def __init__(self, net, lr, gamma):
        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma

    def sample(self, game, player, epsilon=0.1):
        # Exploration
        if torch.rand(1).item() < epsilon:
            valid_moves = game.get_valid_moves(player)
            chosen = valid_moves[torch.randint(0, len(valid_moves), (1,)).item()]
            return chosen

        # Exploitation
        else:
            # Set the network in evaluation mode
            self.net.eval()

            # Disable gradient for higher efficiency
            with torch.no_grad():
                state_tensor = game.get_state().unsqueeze(0)  # Add batch dimension
                q_values = self.net(state_tensor)

            # Greedy action based on Q-values
            return torch.argmax(q_values).item()

    def train(self, batch):
        # Set the network in training mode
        self.net.train()

        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors and compute Q-values
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float)

        q_values = self.net(states)
        q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        next_q_values = self.net(next_states).max(dim=1)[0]

        # Compute target Q-values
        targets = rewards + self.gamma * (1 - dones) * next_q_values

        # Compute loss
        loss = nn.functional.mse_loss(q_values, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
