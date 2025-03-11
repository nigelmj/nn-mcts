from src.alphaZero.pytorch.torchnn import GameZero
from src.games.othello import Othello
from typing import Tuple, List
import numpy as np


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
    "iterations": 200,
    "games_per_iteration": 100,
    "max_iter_per_train_step": 10,
    "num_simulations": 50,
    "batch_size": 256,
    "episode_data_size": 200000,
    "checkpoint_frequency": 100,
    "num_epochs": 10,
    "tournament_games": 40,
    "update_threshold": 0.55,
    "stochastic_threshold": 20,
    "path": "src/alphaZero/othello/models/othello",
    "pt_path": "src/alphaZero/othello/pt_models/othello",
    "keras_path": "src/alphaZero/othello/keras_models/othello"
}

az = OthelloZero()
model = az.build_network(2)
az.training_pipeline(training_config)
