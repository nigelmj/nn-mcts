from src.alphaZero.neural_network import GameZero
from src.games.connect4 import Connect4
from typing import Tuple, List
import numpy as np


class Connect4Zero(GameZero):
    def __init__(self):
        super().__init__(Connect4(), 42)

    def get_move(self, states_len: int, stochastic_threshold: int, improved_policy: np.ndarray) -> Tuple[int, int]:
        if states_len < stochastic_threshold:
            action_index = np.random.choice(len(improved_policy), p=improved_policy)
        else:
            action_index = np.argmax(improved_policy)
        row = int(action_index // 7)
        col = int(action_index % 7)
        return row, col

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
            policy_2d = policy.reshape(6, 7)
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
    "num_simulations": 50,
    "batch_size": 64,
    "replay_buffer_size": 200000,
    "checkpoint_frequency": 100,
    "num_epochs": 10,
    "arena_games": 40,
    "update_threshold": 0.6,
    "stochastic_threshold": 10,
}

az = Connect4Zero()
model = az.build_network(2)
az.training_pipeline(training_config)
