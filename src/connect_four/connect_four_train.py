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
