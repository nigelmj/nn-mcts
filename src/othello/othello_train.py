from typing import List, Tuple

import numpy as np
import torch.multiprocessing as mp

from src.othello.othello_logic import Othello
from src.train_pipeline import GameZero


class OthelloZero(GameZero):
    def __init__(self):
        super().__init__(Othello())

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

            pass_move = policy[-1]
            policy = np.delete(policy, -1)
            policy_2d = policy.reshape(self.game.size1, self.game.size2)

            # Rotations (90°, 180°, 270°)
            for k in range(1, 4):
                rotated_state = np.rot90(state, k=k, axes=(1, 2))

                rotated_policy = np.rot90(policy_2d, k=k).flatten()
                full_policy = np.append(rotated_policy, pass_move)

                augmented_states.append(rotated_state)
                augmented_policies.append(full_policy)
                augmented_values.append(value)

            # Lateral flip
            flipped_state = np.flip(state, axis=2)

            flipped_policy = np.fliplr(policy_2d)
            full_policy = np.append(flipped_policy.flatten(), pass_move)

            augmented_states.append(flipped_state)
            augmented_policies.append(full_policy)
            augmented_values.append(value)

            # Flip + Rotations
            for k in range(1, 4):
                rotated_flipped_state = np.rot90(flipped_state, k=k, axes=(1, 2))

                rotated_flipped_policy = np.rot90(flipped_policy, k=k).flatten()
                full_policy = np.append(rotated_flipped_policy, pass_move)

                augmented_states.append(rotated_flipped_state)
                augmented_policies.append(full_policy)
                augmented_values.append(value)

        return (
            np.array(augmented_states),
            np.array(augmented_policies),
            np.array(augmented_values),
        )


training_config = {
    "iterations": 3000,
    "games_per_iteration": 100,
    "num_simulations": 50,
    "num_steps": 16384,
    "batch_size": 256,
    "replay_buffer_size": 200_000,
    "checkpoint_frequency": 50,
    # "tournament_games": 40,
    # "update_threshold": 0.40,
    "stochastic_threshold": 20,
    "path": "src/othello/models/othello",
    "num_workers": mp.cpu_count() - 1,
    "size1": 8,
    "size2": 8,
    "policy_size": 65,
}

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    print(f"The number of cores available is {mp.cpu_count()}")
    az = OthelloZero()
    model = az.build_network(2)
    az.training_pipeline(training_config)
