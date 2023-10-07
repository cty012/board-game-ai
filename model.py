import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, 81)  # 9x9 board

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 32 * 9 * 9)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
