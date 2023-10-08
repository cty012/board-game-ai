import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 81 * 81)  # 9x9 board

    def forward(self, x_raw):
        # Separate the channels
        empty_channel = (x_raw == -1).float().unsqueeze(1)
        player0_channel = (x_raw == 0).float().unsqueeze(1)
        player1_channel = (x_raw == 1).float().unsqueeze(1)
        x = torch.cat((empty_channel, player0_channel, player1_channel), dim=1)

        x = torch.relu(self.conv1(x))
        x = x.view(-1, 32 * 9 * 9)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample(self, state_tensor, valid_actions):
        q_values = self(state_tensor).squeeze(0)
        masked_q_values = torch.full((81 * 81,), float('-inf'))
        for action in valid_actions:
            masked_q_values[action] = q_values[action]
        return torch.argmax(masked_q_values).item()
