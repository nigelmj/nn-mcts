from typing import List, Tuple

import numpy as np
import torch.multiprocessing as mp

from src.tictactoe.tictactoe_logic import TicTacToe
from src.train_pipeline import GameZero


class TicTacToeZero(GameZero):
    def __init__(self):
        super().__init__(TicTacToe())

    def get_transforms(self, flip_axis: int, rot_axes: Tuple[int]):
        return [
            lambda x: x,
            lambda x: np.rot90(x, k=1, axes=rot_axes),
            lambda x: np.rot90(x, k=2, axes=rot_axes),
            lambda x: np.rot90(x, k=3, axes=rot_axes),
            lambda x: np.flip(x, axis=flip_axis),
            lambda x: np.rot90(np.flip(x, axis=flip_axis), k=1, axes=rot_axes),
            lambda x: np.rot90(np.flip(x, axis=flip_axis), k=2, axes=rot_axes),
            lambda x: np.rot90(np.flip(x, axis=flip_axis), k=3, axes=rot_axes),
        ]

    def augment_data_new(
        self,
        states: List[np.ndarray],
        policies: List[np.ndarray],
        values: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        states = np.asarray(states)
        policies = np.asarray(policies)
        values = np.asarray(values)

        state_transforms = self.get_transforms(3, (2, 3))

        augmented_states = np.concatenate(
            [t(states) for t in state_transforms],
            axis=0,
        )

        policies = policies.reshape(-1, self.game.size1, self.game.size2)
        policy_transforms = self.get_transforms(2, (1, 2))
        augmented_policies = np.concatenate(
            [t(policies) for t in policy_transforms],
            axis=0,
        )
        augmented_policies = augmented_policies.reshape(
            -1, self.game.size1 * self.game.size2
        )

        augmented_values = np.tile(values, 8)
        return (augmented_states, augmented_policies, augmented_values)

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

            policy_2d = policy.reshape(self.game.size1, self.game.size2)

            # Rotations (90°, 180°, 270°)
            for k in range(1, 4):
                rotated_state = np.rot90(state, k=k, axes=(1, 2))
                rotated_policy = np.rot90(policy_2d, k=k).flatten()

                augmented_states.append(rotated_state)
                augmented_policies.append(rotated_policy)
                augmented_values.append(value)

            # Lateral flip
            flipped_state = np.flip(state, axis=2)
            flipped_policy = np.fliplr(policy_2d)

            augmented_states.append(flipped_state)
            augmented_policies.append(flipped_policy.flatten())
            augmented_values.append(value)

            # Flip + Rotations
            for k in range(1, 4):
                rotated_flipped_state = np.rot90(flipped_state, k=k, axes=(1, 2))
                rotated_flipped_policy = np.rot90(flipped_policy, k=k).flatten()

                augmented_states.append(rotated_flipped_state)
                augmented_policies.append(rotated_flipped_policy)
                augmented_values.append(value)

        return (
            np.array(augmented_states),
            np.array(augmented_policies),
            np.array(augmented_values),
        )


training_config = {
    # Data Generation Stage
    "iterations": 10,
    "games_per_iteration": 100,
    "stochastic_threshold": 5,
    "num_simulations": 50,
    # Network Training Stage
    "replay_buffer_size": 5000,
    "num_steps": 1024,
    "batch_size": 256,
    # Tournament Stage
    # "tournament_games": 40,
    # "update_threshold": 0.50,
    # Parallelism and Model Persistence
    "checkpoint_frequency": 5,
    "path": "src/tictactoe/models/TicTacToe",
    "num_workers": mp.cpu_count() - 1,
}

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    az = TicTacToeZero()
    model = az.build_network(2)
    az.training_pipeline(training_config)
