from typing import List, Tuple

import numpy as np
import torch.multiprocessing as mp

from src.hex.hex_logic import Hex
from src.train_pipeline import GameZero


class HexZero(GameZero):
    def __init__(self):
        super().__init__(Hex())

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

            # Rotated
            swap_move = policy[-1]
            policy = np.delete(policy, -1)
            policy_2d = policy.reshape(self.game.size1, self.game.size2)

            rotated_state = np.rot90(state, k=2, axes=(1, 2))
            rotated_policy = np.rot90(policy_2d, k=2).flatten()
            full_policy = np.append(rotated_policy, swap_move)

            augmented_states.append(rotated_state)
            augmented_policies.append(full_policy)
            augmented_values.append(value)

        return (
            np.array(augmented_states),
            np.array(augmented_policies),
            np.array(augmented_values),
        )


training_config = {
    # Data Generation Stage
    "iterations": 30,
    "games_per_iteration": 1000,
    "stochastic_threshold": 60,
    "num_simulations": 500,
    # Network Training Stage
    "replay_buffer_size": 200_000,
    "num_steps": 16_384,
    "batch_size": 256,
    # Tournament Stage
    # "tournament_games": 40,
    # "update_threshold": 0.60,
    # Parallelism and Model Persistence
    "checkpoint_frequency": 5,
    "path": "src/hex/models/hex",
    "num_workers": mp.cpu_count() - 1,
}

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    az = HexZero()
    model = az.build_network(2)
    az.training_pipeline(training_config)
