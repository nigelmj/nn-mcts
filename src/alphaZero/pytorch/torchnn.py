import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
import numpy as np
from src.games.game import Game
from src.alphaZero.apv_node import APVNode
from src.alphaZero.pytorch.apv_mcts import APVMCTS
from src.alphaZero.tournament import Tournament
from random import shuffle
from collections import deque
from typing import List, Tuple, Dict
import time


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
    def __init__(self, game_size1, game_size2, num_channels, policy_size):
        super(AlphaZeroNetwork, self).__init__()

        # Initial conv layer
        self.conv1 = nn.Conv2d(num_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([ResidualBlock(256) for _ in range(5)])

        # Policy head
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * game_size1 * game_size2, policy_size)

        # Value head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(game_size1 * game_size2, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Initial layers
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = torch.log_softmax(self.policy_fc(policy))

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

class GameZero(ABC):
    def __init__(self, game: Game) -> None:
        self.model = None
        self.game = game
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_network(self, num_channels: int) -> nn.Module:
        self.model = AlphaZeroNetwork(
            game_size1=self.game.size1, 
            game_size2=self.game.size2, 
            num_channels=num_channels,
            policy_size=self.game.policy_size
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.001, 
            betas=(0.9, 0.999), 
            eps=1e-8,
            weight_decay=0.001  # L2 regularization
        )

        return self.model

    def generate_games(
        self, num_simulations: int, sampling: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.game.reset()
        states, policies, values = [], [], []

        while not self.game.is_game_over():
            state = self.game.encode_state()

            root = APVNode(self.game.copy(), None, None, 0)
            mcts = APVMCTS(root, self.model, num_simulations, True, sampling)
            improved_policy = mcts.compute_improved_policy()

            states.append(state)
            policies.append(improved_policy)

            action = np.random.choice(len(improved_policy), p=improved_policy)
            self.game.make_move(action)

        result = self.game.get_winner()

        # Values are flipped as they are from the perspective of the current player
        values = [result if i % 2 == 0 else -result for i in range(len(states))]

        return self.augment_data(states, policies, values)

    @abstractmethod
    def augment_data(
        self,
        states: List[np.ndarray],
        policies: List[np.ndarray],
        values: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def train_network(self, training_samples, num_epochs: int, batch_size: int) -> None:
        states, policies, values = zip(*training_samples)

        # Convert to PyTorch tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        policies = torch.tensor(np.array(policies), dtype=torch.float32).to(self.device)
        values = torch.tensor(np.array(values), dtype=torch.float32).view(-1, 1).to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(states, policies, values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        self.model.train()
        history = {'policy_loss': [], 'value_loss': [], 'total_loss': []}

        for epoch in range(num_epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_total_loss = 0.0
            batches = 0

            for batch_states, batch_policies, batch_values in dataloader:
                self.optimizer.zero_grad()

                # Forward pass
                pred_policies, pred_values = self.model(batch_states)

                # Calculate losses
                policy_loss = self.compute_policy_loss(pred_policies, batch_policies)
                value_loss = F.mse_loss(pred_values, batch_values)
                total_loss = policy_loss + value_loss

                # Backward pass and optimize
                total_loss.backward()
                self.optimizer.step()

                # Accumulate losses
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_total_loss += total_loss.item()
                batches += 1

            # Average losses for the epoch
            history['policy_loss'].append(epoch_policy_loss / batches)
            history['value_loss'].append(epoch_value_loss / batches)
            history['total_loss'].append(epoch_total_loss / batches)

            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Policy Loss: {history['policy_loss'][-1]:.4f}, "
                  f"Value Loss: {history['value_loss'][-1]:.4f}, "
                  f"Total Loss: {history['total_loss'][-1]:.4f}")

        return history

    def compute_policy_loss(self, pred_policies, target_policies):
        return -torch.sum(target_policies * pred_policies, dim=1).mean()

    def training_pipeline(self, config: Dict) -> None:
        games_played = 0
        training_data = []

        for _ in range(config["iterations"]):
            # Generate self-play games
            step_time = time.time()
            num_games_per_iteration = config["games_per_iteration"]
            episode_data = deque(maxlen=config['episode_data_size'])
            for _ in range(num_games_per_iteration):
                states, policies, values = self.generate_games(
                    config["num_simulations"],
                    config["stochastic_threshold"] > (games_played % config["games_per_iteration"])
                )
                # print("Games played: ", games_played)

                # Add game data to replay buffer
                for state, policy, value in zip(states, policies, values):
                    episode_data.append((state, policy, value))
                games_played += 1
            print("Games played:", games_played)
            training_data.append(episode_data)

            if len(training_data) > config['max_iter_per_train_step']:
                print("Removing old data")
                training_data.pop(0)

            training_samples = []
            for e in training_data:
                training_samples.extend(e)
            shuffle(training_samples)

            # Clone the current model for comparison
            old_model = AlphaZeroNetwork(
                game_size1=self.game.size1, 
                game_size2=self.game.size2, 
                num_channels=self.model.conv1.in_channels,
                policy_size=self.game.policy_size
            ).to(self.device)
            old_model.load_state_dict(self.model.state_dict())

            self.train_network(
                training_samples,
                num_epochs=config["num_epochs"],
                batch_size=config["batch_size"],
            )

            tournament = Tournament(
                game=self.game,
                curr_model=old_model,
                new_model=self.model,
                num_games=config["tournament_games"],
                threshold=config["update_threshold"],
                num_simulations=config["num_simulations"],
            )

            if not tournament.cross_threshold():
                print("New model rejected; reverting to the old model.")
                self.model.load_state_dict(old_model.state_dict())
            else:
                print("New model accepted based on tournament evaluation.")

            # Save checkpoint
            if games_played % config["checkpoint_frequency"] == 0:
                torch.save(self.model.state_dict(), f"{config['pt_path']}_checkpoint_{games_played}.pt")
            print("Time taken for training step:", time.time()-step_time)
            step_time = time.time()
