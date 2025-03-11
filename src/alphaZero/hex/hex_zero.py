from src.alphaZero.pytorch.torchnn import GameZero
from src.games.hex import Hex
from typing import Tuple, List
import numpy as np

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
    "iterations": 150,
    "games_per_iteration": 100,
    "max_iter_per_train_step": 10,
    "num_simulations": 50,
    "batch_size": 128,
    "episode_data_size": 300000,
    "checkpoint_frequency": 100,
    "num_epochs": 10,
    "tournament_games": 40,
    "update_threshold": 0.55,
    "stochastic_threshold": 25,
    "path": "src/alphaZero/hex/models/hex",
    "pt_path": "src/alphaZero/hex/pt_models/hex",
    "keras_path": "src/alphaZero/hex/keras_models/hex"
}

az = HexZero()
model = az.build_network(2)
az.training_pipeline(training_config)
