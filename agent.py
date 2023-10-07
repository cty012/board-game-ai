import torch


class QLearningAgent:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_action(self, game, player, epsilon=0.1):
        if torch.rand(1).item() < epsilon:  # Exploration
            valid_moves = game.get_valid_moves(player)
            chosen = valid_moves[torch.randint(0, len(valid_moves), (1,)).item()]
            return chosen  # Random action
        else:  # Exploitation
            with torch.no_grad():
                state_tensor = game.get_state().type(torch.float32).unsqueeze(0)  # Add batch dimension
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()  # Greedy action based on Q-values

    def train(self, state, action, reward, next_state):
        # Implement training logic
        # TODO
        pass
