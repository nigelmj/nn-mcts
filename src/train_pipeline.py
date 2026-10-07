import os
import pickle
import time
from abc import ABC, abstractmethod
from random import sample
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.checkpoint import load_checkpoint, save_checkpoint
from src.game import Game

# from src.tournament import Tournament
from src.inference_worker import InferenceWorker
from src.mcts import MCTS, apply_temperature, temperature_for_move
from src.neural_network import AlphaZeroNetwork
from src.node import Node
from src.parallel_utils import generation_worker


class ReplayMemory:
    def __init__(self, max_size: int) -> None:
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    @staticmethod
    def _compact(obj: Tuple[np.ndarray, np.ndarray, int]) -> Tuple[np.ndarray, np.ndarray, int]:
        """Store states as int8 and policies as float32.

        encode_state produces int64 boards and the search produces float64 visit
        distributions, about 2.9 KB per 11x11 position; this is ~4x smaller in memory and
        in the saved buffer. Training converts both to float32 tensors anyway, so nothing
        downstream changes. A state that is not small integers is kept as float32.
        """
        state, policy, value = obj
        state = np.asarray(state)
        if (np.issubdtype(state.dtype, np.integer) or state.dtype == np.bool_) and (
            state.size == 0 or (state.min() >= -128 and state.max() <= 127)
        ):
            state = state.astype(np.int8)
        else:
            state = state.astype(np.float32)
        return state, np.asarray(policy, dtype=np.float32), value

    def append(self, obj: Tuple[np.ndarray]) -> None:
        self.buffer[self.index] = self._compact(obj)
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def extend(self, objs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        for obj in objs:
            self.append(obj)

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray]]:
        indices = sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]

    def entries(self) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """The stored positions, oldest first."""
        if self.size < self.max_size:
            return self.buffer[: self.size]
        return self.buffer[self.index :] + self.buffer[: self.index]

    def save(self, save_path: str) -> None:
        # Temporary file + rename, so a crash mid-write cannot destroy the previous save.
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = f"{save_path}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(
                {
                    "buffer": self.buffer,
                    "max_size": self.max_size,
                    "index": self.index,
                    "size": self.size,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(tmp, save_path)

    @classmethod
    def load(cls, load_path: str, max_size: int) -> "ReplayMemory":
        """Load a saved buffer into a new one of size max_size.

        The size may differ from the saved one: positions are re-added oldest first, so
        a smaller buffer keeps the most recent ones. Buffers saved before compaction
        (int64/float64) are compacted on the way in.
        """
        with open(load_path, "rb") as f:
            saved = pickle.load(f)
        old = cls.__new__(cls)
        old.buffer = saved["buffer"]
        old.max_size = saved["max_size"]
        old.index = saved["index"]
        old.size = saved["size"]
        memory = cls(max_size)
        memory.extend(old.entries())
        return memory


def lr_for_iteration(schedule: Sequence[Tuple[int, float]], iteration: int) -> float:
    """Learning rate for a 0-based global iteration from a [(from_iteration, lr), ...]
    step schedule, e.g. [(0, 1e-3), (90, 3e-4), (125, 1e-4)].

    `iteration` counts across jobs (it is restored from the checkpoint on resume), so the
    schedule continues where the previous job stopped. Nothing about the schedule needs
    saving: the rate is recomputed from the iteration every time.
    """
    if not schedule:
        raise ValueError("empty lr_schedule; omit the key for a constant learning rate")
    ordered = sorted(schedule, key=lambda entry: entry[0])
    lr = ordered[0][1]
    for from_iteration, value in ordered:
        if iteration >= from_iteration:
            lr = value
        else:
            break
    return float(lr)


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
        temperature_schedule: Sequence[Tuple[int, float]],
        req_q: mp.Queue,
        resp_q: mp.Queue,
        wid: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        states, policies, values = [], [], []

        self.game.reset()
        move_count = 0
        root = Node(self.game.copy(), None, -1, 0)

        while not self.game.is_game_over():
            state = self.game.encode_state()

            mcts = MCTS(
                root,
                num_simulations,
                True,
                True,
                request_queue=req_q,
                response_queue=resp_q,
                wid=wid,
            )
            improved_policy = mcts.compute_improved_policy()

            # The target is the full visit distribution, at every ply. Temperature
            # applies only to picking the move actually played, so late-game positions
            # still contribute everything the search knows about them.
            states.append(state)
            policies.append(improved_policy)

            temperature = temperature_for_move(temperature_schedule, move_count)
            play_policy = apply_temperature(improved_policy, temperature)
            action = int(np.random.choice(len(play_policy), p=play_policy))
            self.game.make_move(action)

            root = root.children[action]
            root.parent = None

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
        temperature_schedule: Sequence[Tuple[int, float]],
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
                    temperature_schedule,
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

    def clean_weight_keys(self, state_dict: Dict) -> Dict:
        new_state_dict = {}
        for k, v in state_dict.items():
            # Strip the prefix dynamically
            clean_k = k.replace("_orig_mod.", "")
            new_state_dict[clean_k] = v.detach().cpu()

        return new_state_dict

    def save_training_state(
        self, path: str, iteration: int, games_played: int, config: Dict
    ) -> None:
        """Everything needed to continue the run, in one file (see src/checkpoint.py).
        The replay buffer is saved separately, at the end of the job."""
        save_checkpoint(
            path,
            self.clean_weight_keys(self.model.state_dict()),
            self.optimizer.state_dict(),
            iteration,
            games_played,
            config,
        )

    def training_pipeline(self, config: Dict) -> None:
        """Run config["iterations"] iterations of self-play + training.

        Optional config keys:
            resume_from    checkpoint to continue from (model, optimizer, iteration count,
                           games played). None or missing = start fresh.
            resume_buffer  also load config["buffer_path"] when resuming (default True).
            lr_schedule    [(from_iteration, lr), ...] keyed on the global iteration.
                           Missing = keep the optimizer's learning rate.
            save_latest    also write <models_path>_latest.pt after every iteration
                           (default True). Overwritten each time; cheap (~10 MB).
        """
        assert self.model is not None

        temperature_schedule = config["temperature_schedule"]
        lr_schedule = config.get("lr_schedule")
        models_path = config["models_path"]
        buffer_path = config["buffer_path"]
        print(f"Self-play temperature schedule: {temperature_schedule}")
        print(f"Learning-rate schedule: {lr_schedule if lr_schedule else 'constant'}")

        # ---- start fresh, or continue a previous run --------------------------------
        start_iteration, games_played = 0, 0
        resume_from = config.get("resume_from")
        if resume_from:
            checkpoint = load_checkpoint(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if checkpoint["optimizer_state_dict"] is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            else:
                print("Checkpoint has no optimizer state (weights-only file); "
                      "the optimizer starts fresh")
            if checkpoint["iteration"] is None:
                print("Checkpoint has no iteration count (weights-only file); "
                      "counting from 0 again")
            start_iteration = checkpoint["iteration"] or 0
            games_played = checkpoint["games_played"] or 0
            print(f"Resumed from {resume_from} "
                  f"({start_iteration} iterations, {games_played:,} games completed)")

        if resume_from and config.get("resume_buffer", True) and os.path.exists(buffer_path):
            replay_buffer = ReplayMemory.load(buffer_path, config["replay_buffer_size"])
            print(f"Loaded replay buffer from {buffer_path}: {replay_buffer.size:,} positions")
        else:
            replay_buffer = ReplayMemory(max_size=config["replay_buffer_size"])
            if resume_from:
                print(f"No replay buffer loaded (looked for {buffer_path}); starting empty")

        ctx = mp.get_context("spawn")
        request_queue = ctx.Queue()
        response_queues_dict = {}

        response_queues_dict["main"] = ctx.Queue()
        response_queues_dict["model"] = ctx.Queue()
        for wid in range(config["num_workers"]):
            response_queues_dict[wid] = ctx.Queue()

        # After any resume, so self-play starts from the restored weights.
        state_dict = self.clean_weight_keys(self.model.state_dict())

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

        # `iteration` is 0-based and global: the first iteration of a resumed job is the
        # number of iterations already completed. Checkpoint names use iteration + 1.
        last_iteration = start_iteration + config["iterations"] - 1
        for iteration in range(start_iteration, last_iteration + 1):
            step_time = time.time()
            num_games_per_iteration = config["games_per_iteration"]

            episode_data = self.parallel_generate(
                num_games_per_iteration,
                config["num_simulations"],
                temperature_schedule,
                request_queue,
                response_queues_dict,
                config["num_workers"],
            )

            games_played += num_games_per_iteration - (
                num_games_per_iteration % config["num_workers"]
            )

            print(f"Iteration {iteration + 1}")
            print(f"Games played is {games_played}")
            print(f"Length of episode data is {len(episode_data):,}")
            print(f"Time take for episode is {time.time() - step_time:.4f}", flush=True)

            replay_buffer.extend(episode_data)

            if (iteration + 1) % 10 == 0:
                print(
                    f"Length of training samples for iteration {iteration + 1} is {replay_buffer.size}"
                )

            if lr_schedule:
                lr = lr_for_iteration(lr_schedule, iteration)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
            print(f"Learning rate {self.optimizer.param_groups[0]['lr']:g}")

            self.train_network(
                replay_buffer,
                num_steps=config["num_steps"],
                batch_size=config["batch_size"],
            )

            new_state_dict = self.clean_weight_keys(self.model.state_dict())
            request_queue.put(("update_weights", iteration, new_state_dict))
            response = response_queues_dict["model"].get()
            print(response)

            completed = iteration + 1
            if completed % config["checkpoint_frequency"] == 0 or iteration == last_iteration:
                print(f"Saving iteration {completed} checkpoint...")
                self.save_training_state(
                    f"{models_path}_checkpoint_{completed}.pt", completed, games_played, config
                )
            if config.get("save_latest", True):
                self.save_training_state(
                    f"{models_path}_latest.pt", completed, games_played, config
                )
            print(
                f"Time taken for training step is {time.time() - step_time:.4f}",
                flush=True,
                end="\n\n",
            )

        request_queue.put("shutdown")
        response = response_queues_dict["main"].get()
        print(response)
        self.inference_worker.join()

        save_time = time.time()
        replay_buffer.save(save_path=buffer_path)
        print(f"Saved replay buffer ({replay_buffer.size:,} positions) to {buffer_path} "
              f"in {time.time() - save_time:.1f}s")
