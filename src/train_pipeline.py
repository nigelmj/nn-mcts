import time
from abc import ABC, abstractmethod
from random import sample
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.game import Game

# from src.tournament import Tournament
from src.inference_worker import InferenceWorker
from src.mcts import MCTS
from src.neural_network import AlphaZeroNetwork
from src.node import Node
from src.parallel_utils import generation_worker


class ReplayMemory:
    def __init__(self, max_size: int) -> None:
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj: Tuple[np.ndarray]) -> None:
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def extend(self, objs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        for obj in objs:
            self.append(obj)

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray]]:
        indices = sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]


class GameZero(ABC):
    def __init__(self, game: Game) -> None:
        self.model = None
        self.game = game
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def build_network(self, num_channels: int) -> nn.Module:
        self.model = AlphaZeroNetwork(
            game_size1=self.game.size1,
            game_size2=self.game.size2,
            num_channels=num_channels,
            policy_size=self.game.policy_size,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.001,  # L2 regularization
        )

        return self.model

    def generate_games(
        self,
        num_simulations: int,
        threshold: int,
        req_q: mp.Queue,
        resp_q: mp.Queue,
        wid: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        states, policies, values = [], [], []

        self.game.reset()
        move_count = 0

        while not self.game.is_game_over():
            state = self.game.encode_state()

            sampling = move_count < threshold
            root = Node(self.game.copy(), None, -1, 0)
            mcts = MCTS(
                root,
                num_simulations,
                True,
                sampling,
                True,
                request_queue=req_q,
                response_queue=resp_q,
                wid=wid,
            )
            improved_policy = mcts.compute_improved_policy()

            states.append(state)
            policies.append(improved_policy)

            action = np.random.choice(len(improved_policy), p=improved_policy)
            self.game.make_move(action)

            move_count += 1

        result = self.game.get_winner()

        # Values are flipped as they are from the perspective of the current player
        values = [result if i % 2 == 0 else -result for i in range(len(states))]

        return np.array(states), np.array(policies), np.array(values)

    @abstractmethod
    def augment_data(
        self,
        states: List[np.ndarray],
        policies: List[np.ndarray],
        values: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def parallel_generate(
        self,
        total_games: int,
        num_simulations: int,
        threshold: int,
        req_q: mp.Queue,
        resp_q_dict: dict[int, mp.Queue],
        num_workers: int,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        games_per_worker = total_games // num_workers

        # start = time.time()
        game_cls = self.__class__
        ctx = mp.get_context("spawn")
        processes = []
        results_queue = ctx.Queue()

        for wid in range(num_workers):
            p = ctx.Process(
                target=generation_worker,
                args=(
                    req_q,
                    resp_q_dict[wid],
                    game_cls,
                    games_per_worker,
                    num_simulations,
                    threshold,
                    results_queue,
                    wid,
                ),
            )
            processes.append(p)
            p.start()

        overall_results = []
        for i in range(num_workers):
            worker_result = results_queue.get()
            overall_results.extend(worker_result)
        for p in processes:
            p.join()
        return overall_results

    def train_network(self, replay_buffer, num_steps: int, batch_size: int) -> None:
        assert self.model is not None

        # Training loop
        self.model.train()
        history = {"policy_loss": [], "value_loss": []}

        train_step_policy_loss = 0.0
        train_step_value_loss = 0.0

        for train_step in range(num_steps):
            # Sampling a subset, augmenting it with additional data
            # and resampling to ensure batch size remains the same
            sample_data = replay_buffer.sample(batch_size=batch_size)
            sample_states, sample_policies, sample_values = zip(*sample_data)
            augmented_states, augmented_policies, augmented_values = self.augment_data(
                sample_states, sample_policies, sample_values
            )

            batch_indices = np.random.randint(0, len(augmented_states), size=batch_size)
            batch_states = augmented_states[batch_indices]
            batch_policies = augmented_policies[batch_indices]
            batch_values = augmented_values[batch_indices]

            batch_states = torch.from_numpy(batch_states).float().to(self.device)
            batch_policies = torch.from_numpy(batch_policies).float().to(self.device)
            batch_values = (
                torch.from_numpy(batch_values).float().view(-1, 1).to(self.device)
            )

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
            train_step_policy_loss += policy_loss.item()
            train_step_value_loss += value_loss.item()

        # Average losses for the training steps
        history["policy_loss"].append(train_step_policy_loss / num_steps)
        history["value_loss"].append(train_step_value_loss / num_steps)

        print(
            f"Training Step, "
            f"Policy Loss: {history['policy_loss'][-1]:.4f}, "
            f"Value Loss: {history['value_loss'][-1]:.4f}"
        )

    def compute_policy_loss(self, pred_policies, target_policies):
        return -torch.sum(target_policies * pred_policies, dim=1).mean()

    def training_pipeline(self, config: Dict) -> None:
        assert self.model is not None

        games_played = 0
        replay_buffer = ReplayMemory(max_size=config["replay_buffer_size"])

        ctx = mp.get_context("spawn")
        request_queue = ctx.Queue()
        response_queues_dict = {}

        response_queues_dict["main"] = ctx.Queue()
        response_queues_dict["model"] = ctx.Queue()
        for wid in range(config["num_workers"]):
            response_queues_dict[wid] = ctx.Queue()

        state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        self.inference_worker = InferenceWorker(
            state_dict,
            request_queue,
            response_queues_dict,
            self.device,
            config["num_workers"],
            self.game.size1,
            self.game.size2,
            self.game.policy_size,
        )

        self.inference_worker.start()

        for iteration in range(config["iterations"]):
            step_time = time.time()
            num_games_per_iteration = config["games_per_iteration"]

            episode_data = self.parallel_generate(
                num_games_per_iteration,
                config["num_simulations"],
                config["stochastic_threshold"],
                request_queue,
                response_queues_dict,
                config["num_workers"],
            )

            games_played += num_games_per_iteration - (
                num_games_per_iteration % config["num_workers"]
            )

            print(f"Games played is {games_played}")
            print(f"Length of episode data is {len(episode_data):,}")
            print(f"Time take for episode is {time.time() - step_time:.4f}", flush=True)

            replay_buffer.extend(episode_data)

            if (iteration + 1) % 10 == 0:
                print(
                    f"Length of training samples for iteration {iteration + 1} is {replay_buffer.size}"
                )

            self.train_network(
                replay_buffer,
                num_steps=config["num_steps"],
                batch_size=config["batch_size"],
            )

            new_state_dict = {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            }
            request_queue.put(("update_weights", iteration, new_state_dict))
            response = response_queues_dict["model"].get()
            print(response)

            # Save checkpoint
            if (iteration + 1) % config["checkpoint_frequency"] == 0:
                print(f"Saving iteration {iteration + 1} model...")
                torch.save(
                    self.model.state_dict(),
                    f"{config['path']}_checkpoint_{iteration + 1}.pt",
                )
            print(
                f"Time taken for training step is {time.time() - step_time:.4f}",
                flush=True,
                end="\n\n",
            )
            step_time = time.time()

        request_queue.put("shutdown")
        response = response_queues_dict["main"].get()
        print(response)
        self.inference_worker.join()
