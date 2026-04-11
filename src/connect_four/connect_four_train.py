import torch.multiprocessing as mp
from src.train_pipeline import GameZero
from src.connect_four.connect_four_logic import ConnectFour
from src.parallel_utils import worker_generate_games
from typing import Tuple, List
import numpy as np


class ConnectFourZero(GameZero):
    def __init__(self):
        super().__init__(ConnectFour())

    def augment_data(
        self,
        states: List[np.ndarray],
        policies: List[np.ndarray],
        values: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        augmented_states = []
        augmented_policies = []
        augmented_values = []

        for state, policy, value in zip(states, policies, values):
            # Original
            augmented_states.append(state)
            augmented_policies.append(policy)
            augmented_values.append(value)

            # Lateral flip
            policy_2d = policy.reshape(self.game.size1, self.game.size2)
            flipped_state = np.flip(state, axis=2)
            flipped_policy = np.fliplr(policy_2d).flatten()

            augmented_states.append(flipped_state)
            augmented_policies.append(flipped_policy)
            augmented_values.append(value)

        return (
            np.array(augmented_states),
            np.array(augmented_policies),
            np.array(augmented_values),
        )

    def parallel_generate(self,
        total_games: int,
        num_simulations: int,
        threshold: int,
        req_q: mp.Queue,
        resp_q_dict: dict[int, mp.Queue],
        num_workers: int
    ) -> List:
        games_per_worker = total_games // num_workers

        ctx = mp.get_context("spawn")
        processes = []
        results_queue = mp.Queue()

        for wid in range(num_workers):
            p = ctx.Process(
                target=worker_generate_games,
                args=(req_q, resp_q_dict[wid], ConnectFourZero, games_per_worker, num_simulations, threshold, results_queue, wid)
            )
            processes.append(p)
            p.start()

        overall_results = []
        for _ in range(num_workers):
            worker_result = results_queue.get()
            print("Worker result length of result:", len(worker_result))
            overall_results.extend(worker_result)

        for p in processes:
            p.join()
        return overall_results


training_config = {
    "iterations": 100,
    "games_per_iteration": 200,
    "max_iter_per_train_step": 20,
    "num_simulations": 50,
    "batch_size": 32,
    "episode_data_size": 60000,
    "checkpoint_frequency": 200,
    "num_epochs": 10,
    "tournament_games": 40,
    "update_threshold": 0.60,
    "stochastic_threshold": 20,
    "path": "src/connect4/models/connect4",
    "num_workers": mp.cpu_count() - 1,
    "size1": 6,
    "size2": 7,
    "policy_size": 42
}

if __name__ == "__main__":
    az = ConnectFourZero()
    model = az.build_network(2)
    az.training_pipeline(training_config)
