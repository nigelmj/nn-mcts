import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AlphaZeroNetwork(nn.Module):
    def __init__(self, game_size1, game_size2, policy_size, num_channels=2):
        super(AlphaZeroNetwork, self).__init__()

        # Initial conv layer
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(8)])

        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * game_size1 * game_size2, policy_size)

        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(game_size1 * game_size2, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Initial layers
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = torch.flatten(policy, start_dim=1)
        policy = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = torch.flatten(value, start_dim=1)
        value = F.relu(self.value_fc1(value))
        value = self.value_fc2(value)

        return F.log_softmax(policy, dim=1), torch.tanh(value)

    def predict(self, state_tensor):
        with torch.no_grad():
            self.eval()
            policy_log, value_tensor = self(state_tensor)
            policy = torch.exp(policy_log).cpu().numpy()
            value = value_tensor.cpu().numpy()[0][0]
            return policy, value

    def predict_batch(self, state_batch_tensor):
        with torch.no_grad():
            self.eval()
            policy_log, value_tensor = self(state_batch_tensor)
            policies = torch.exp(policy_log).cpu().numpy()
            values = value_tensor.cpu().numpy().flatten()
            return policies, values
