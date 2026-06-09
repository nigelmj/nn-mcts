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
        states = np.asarray(states)
        policies = np.asarray(policies)
        values = np.asarray(values)

        rotated_states = np.rot90(states, k=2, axes=(2, 3))

        swap_move_policies = policies[:, -1]
        tile_move_policies = policies[:, :-1]
        tile_move_policies = tile_move_policies.reshape(
            -1, self.game.size1, self.game.size2
        )
        rotated_policies = np.rot90(tile_move_policies, k=2, axes=(1, 2)).reshape(
            -1, self.game.size1 * self.game.size2
        )
        full_policies = np.concatenate(
            [rotated_policies, swap_move_policies[:, None]], axis=1
        )

        augmented_states = np.concatenate([states, rotated_states], axis=0)
        augmented_policies = np.concatenate([policies, full_policies], axis=0)
        augmented_values = np.concatenate([values, values], axis=0)
        return (augmented_states, augmented_policies, augmented_values)


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
