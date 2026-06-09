from typing import List, Tuple

import numpy as np
import torch.multiprocessing as mp

from src.connect_four.connect_four_logic import ConnectFour
from src.train_pipeline import GameZero


class ConnectFourZero(GameZero):
    def __init__(self):
        super().__init__(ConnectFour())

    def augment_data(
        self,
        states: List[np.ndarray],
        policies: List[np.ndarray],
        values: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        states = np.asarray(states)
        policies = np.asarray(policies)
        values = np.asarray(values)

        flipped_states = np.flip(states, axis=3)
        policies_2d = policies.reshape(-1, self.game.size1, self.game.size2)
        flipped_policies = np.flip(policies_2d, axis=2)
        flipped_policies = flipped_policies.reshape(
            -1, self.game.size1 * self.game.size2
        )

        augmented_states = np.concatenate([states, flipped_states], axis=0)
        augmented_policies = np.concatenate([policies, flipped_policies], axis=0)
        augmented_values = np.concatenate([values, values], axis=0)
        return (augmented_states, augmented_policies, augmented_values)


training_config = {
    # Data Generation Stage
    "iterations": 3000,
    "games_per_iteration": 100,
    "stochastic_threshold": 10,
    "num_simulations": 100,
    # Network Training Stage
    "replay_buffer_size": 100_000,
    "num_steps": 8192,
    "batch_size": 256,
    # Tournament Stage
    # "tournament_games": 40,
    # "update_threshold": 0.60,
    # Parallelism and Model Persistence
    "checkpoint_frequency": 500,
    "path": "src/connect_four/models/connect_four",
    "num_workers": mp.cpu_count() - 1,
}

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    az = ConnectFourZero()
    model = az.build_network(2)
    az.training_pipeline(training_config)
