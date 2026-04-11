import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
import numpy as np
from src.game import Game
from src.node import Node
from src.mcts import MCTS
from src.neural_network import AlphaZeroNetwork
from src.tournament import Tournament
from src.inference_worker import InferenceWorker
from random import shuffle
from collections import deque
from typing import List, Tuple, Dict
import time


class GameZero(ABC):
    def __init__(self, game: Game) -> None:
        self.model = None
        self.game = game
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps")

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
        self, num_simulations: int, threshold: int, req_q: mp.Queue, resp_q: mp.Queue, wid: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        states, policies, values = [], [], []

        self.game.reset()
        root = Node(self.game.copy(), None, None, 0)
        move_count = 0

        while not self.game.is_game_over():
            state = self.game.encode_state()

            sampling = move_count < threshold
            mcts = MCTS(root, num_simulations, True, sampling, req_q, resp_q, wid)
            improved_policy = mcts.compute_improved_policy()

            states.append(state)
            policies.append(improved_policy)

            action = np.random.choice(len(improved_policy), p=improved_policy)
            self.game.make_move(action)

            if action in root.children:
                root = root.children[action]
                root.parent = None
            move_count += 1

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

    @abstractmethod
    def parallel_generate(self,
        total_games: int,
        num_simulations: int,
        threshold: int,
        req_q: mp.Queue,
        resp_q_dict: dict[int, mp.Queue],
        num_workers: int
    ) -> List:
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
        history = {'policy_loss': [], 'value_loss': []}

        for epoch in range(num_epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
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
                batches += 1

            # Average losses for the epoch
            history['policy_loss'].append(epoch_policy_loss / batches)
            history['value_loss'].append(epoch_value_loss / batches)

            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Policy Loss: {history['policy_loss'][-1]:.4f}, "
                  f"Value Loss: {history['value_loss'][-1]:.4f}", flush=True)

    def compute_policy_loss(self, pred_policies, target_policies):
        return -torch.sum(target_policies * pred_policies, dim=1).mean()

    def training_pipeline(self, config: Dict) -> None:
        games_played = 0
        training_data = []

        request_queue = mp.Queue()
        response_queues_dict = {}

        for wid in range(config["num_workers"]):
            response_queues_dict[wid] = mp.Queue()

        state_dict = {
            k: v.detach().cpu()
            for k, v in self.model.state_dict().items()
        }

        self.inference_worker = InferenceWorker(
            state_dict,
            request_queue,
            response_queues_dict,
            self.device,
            config["num_workers"],
            config["size1"],
            config["size2"],
            config["policy_size"]
        )

        self.inference_worker.start()
        print(f"Queue for training pipeline is {id(request_queue)}")

        for _ in range(config["iterations"]):
            step_time = time.time()
            num_games_per_iteration = config["games_per_iteration"]
            episode_data = deque(maxlen=config['episode_data_size'])

            result = self.parallel_generate(
                num_games_per_iteration,
                config["num_simulations"],
                config["stochastic_threshold"],
                request_queue,
                response_queues_dict,
                config["num_workers"]
            )
            episode_data.extend(result)

            # games_played += num_games_per_iteration

            # print("Games played:", games_played, flush=True)
            # print("Length of episode data:", len(episode_data))
            # print("Time take for episode", time.time() - step_time)

            # training_data.append(episode_data)

            # if len(training_data) > config['max_iter_per_train_step']:
            #     print("Removing old data")
            #     training_data.pop(0)

            # training_samples = []
            # for e in training_data:
            #     training_samples.extend(e)
            # shuffle(training_samples)

            # # Clone the current model for comparison
            # old_model = AlphaZeroNetwork(
            #     game_size1=self.game.size1,
            #     game_size2=self.game.size2,
            #     num_channels=self.model.conv1.in_channels,
            #     policy_size=self.game.policy_size
            # ).to(self.device)
            # old_model.load_state_dict(self.model.state_dict())

            # self.train_network(
            #     training_samples,
            #     num_epochs=config["num_epochs"],
            #     batch_size=config["batch_size"],
            # )

            # tournament = Tournament(
            #     game=self.game,
            #     curr_model=old_model,
            #     new_model=self.model,
            #     num_games=config["tournament_games"],
            #     threshold=config["update_threshold"],
            #     num_simulations=config["num_simulations"],
            # )

            # if not tournament.cross_threshold():
            #     print("New model rejected; reverting to the old model.")
            #     self.model.load_state_dict(old_model.state_dict())
            # else:
            #     print("New model accepted based on tournament evaluation.")

            # # Save checkpoint
            # if games_played % config["checkpoint_frequency"] == 0:
            #     torch.save(self.model.state_dict(), f"{config['path']}_checkpoint_{games_played}.pt")
            # print("Time taken for training step:", time.time()-step_time, flush=True)
            # step_time = time.time()
