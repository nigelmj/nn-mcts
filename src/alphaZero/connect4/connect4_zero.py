from src.alphaZero.pytorch.torchnn import GameZero
from src.games.connect4 import Connect4
from typing import Tuple, List
import numpy as np


class Connect4Zero(GameZero):
    def __init__(self):
        super().__init__(Connect4())

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
    "iterations": 100,
    "games_per_iteration": 100,
    "max_iter_per_train_step": 10,
    "num_simulations": 50,
    "batch_size": 64,
    "episode_data_size": 60000,
    "checkpoint_frequency": 100,
    "num_epochs": 10,
    "tournament_games": 40,
    "update_threshold": 0.55,
    "stochastic_threshold": 10,
    "path": "src/alphaZero/connect4/models/connect4",
    "pt_path": "src/alphaZero/connect4/pt_models/connect4",
    "keras_path": "src/alphaZero/connect4/keras_models/connect4"
}

az = Connect4Zero()
model = az.build_network(2)
az.training_pipeline(training_config)
