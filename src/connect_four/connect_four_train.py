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

    def parallel_generate(self, total_games, num_simulations, threshold, num_workers):
        model_state_dict = self.model.state_dict()

        games_per_worker = total_games // num_workers

        ctx = mp.get_context("spawn")
        with ctx.Pool(num_workers) as pool:
            results = pool.starmap(
                worker_generate_games,
                [
                    (model_state_dict, ConnectFourZero, games_per_worker, num_simulations, threshold)
                    for _ in range(num_workers)
                ]
            )

        overall_results = []
        for worker_result in results:
            print("Worker result length of result:", len(worker_result))
            overall_results.extend(worker_result)
        return overall_results


training_config = {
    "iterations": 100,
    "games_per_iteration": 200,
    "max_iter_per_train_step": 20,
    "num_simulations": 25,
    "batch_size": 32,
    "episode_data_size": 60000,
    "checkpoint_frequency": 200,
    "num_epochs": 10,
    "tournament_games": 40,
    "update_threshold": 0.60,
    "stochastic_threshold": 20,
    "path": "src/connect4/models/connect4",
    "num_workers": 7
}

if __name__ == "__main__":
    az = ConnectFourZero()
    model = az.build_network(2)
    az.training_pipeline(training_config)
