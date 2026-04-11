import torch.multiprocessing as mp
from src.train_pipeline import GameZero
from src.othello.othello_logic import Othello
from src.parallel_utils import worker_generate_games
from typing import Tuple, List
import numpy as np
import time


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

    def parallel_generate(self,
        total_games: int,
        num_simulations: int,
        threshold: int,
        req_q: mp.Queue,
        resp_q_dict: dict[int, mp.Queue],
        num_workers: int
    ) -> List:
        games_per_worker = total_games // num_workers

        start = time.time()
        ctx = mp.get_context("spawn")
        processes = []
        results_queue = mp.Queue()

        for wid in range(num_workers):
            p = ctx.Process(
                target=worker_generate_games,
                args=(req_q, resp_q_dict[wid], OthelloZero, games_per_worker, num_simulations, threshold, results_queue, wid)
            )
            processes.append(p)
            p.start()

        overall_results = []
        for _ in range(num_workers):
            worker_result = results_queue.get()
            overall_results.extend(worker_result)
        print(f"Total parallelised generation time is {time.time() - start}")
        for p in processes:
            p.join()
        return overall_results


training_config = {
    "iterations": 100,
    "games_per_iteration": 200,
    "max_iter_per_train_step": 10,
    "num_simulations": 50,
    "batch_size": 32,
    "episode_data_size": 200000,
    "checkpoint_frequency": 100,
    "num_epochs": 10,
    "tournament_games": 40,
    "update_threshold": 0.40,
    "stochastic_threshold": 30,
    "path": "src/othello/models/othello",
    "num_workers": mp.cpu_count() - 1,
    "size1": 8,
    "size2": 8,
    "policy_size": 65
}

if __name__ == "__main__":
    az = OthelloZero()
    model = az.build_network(2)
    az.training_pipeline(training_config)
