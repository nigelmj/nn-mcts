import torch.multiprocessing as mp
from src.tictactoe.tictactoe_logic import TicTacToe
from src.train_pipeline import GameZero
from src.parallel_utils import worker_generate_games
from typing import Tuple, List
import numpy as np


class TicTacToeZero(GameZero):
    def __init__(self):
        super().__init__(TicTacToe())

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

    def parallel_generate(self, total_games, num_simulations, threshold, num_workers):
        model_state_dict = self.model.state_dict()

        games_per_worker = total_games // num_workers

        ctx = mp.get_context("spawn")
        with ctx.Pool(num_workers) as pool:
            results = pool.starmap(
                worker_generate_games,
                [
                    (model_state_dict, TicTacToeZero, games_per_worker, num_simulations, threshold)
                    for _ in range(num_workers)
                ]
            )

        overall_results = []
        for worker_result in results:
            print("Worker result length of result:", len(worker_result))
            overall_results.extend(worker_result)
        return overall_results


training_config = {
    "iterations": 10,
    "games_per_iteration": 100,
    "max_iter_per_train_step": 5,
    "num_simulations": 50,
    "batch_size": 32,
    "episode_data_size": 50000,
    "checkpoint_frequency": 50,
    "num_epochs": 10,
    "tournament_games": 40,
    "update_threshold": 0.50,
    "stochastic_threshold": 15,
    "path": "src/tictactoe/models/TicTacToe",
    "num_workers": 8
}

if __name__ == "__main__":
    az = TicTacToeZero()
    model = az.build_network(2)
    az.training_pipeline(training_config)
