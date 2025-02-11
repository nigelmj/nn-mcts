from src.games.game import Game
from src.alphaZero.apv_node import APVNode
import tensorflow as tf
from tensorflow.keras.models import Model
import random
import numpy as np
from typing import List, Tuple


class APVMCTS:
    def __init__(self,
        root: APVNode,
        model: Model,
        num_simulations: int,
        policy_size: int,
        training: bool,
    ) -> None:
        self.root = root
        self.model = model
        self.num_simulations = num_simulations
        self.policy_size = policy_size
        self.training = training

    def compute_improved_policy(self) -> np.ndarray:
        for _ in range(self.num_simulations):
            search_path = []
            node = self._selection(self.root, search_path)
            result = self._expansion(node, search_path)
            self._backpropagation(result, search_path)

        total_visits = sum(child.visit_count for child in self.root.children)
        policy = np.zeros(self.policy_size)

        for child in self.root.children:
            if child.action:
                if child.action == (-1, -1):
                    action_idx = -1
                else:
                    action_idx = child.action[0] * self.root.game.size2 + child.action[1]
                policy[action_idx] = child.visit_count / total_visits

        policy /= np.sum(policy)
        return policy

    def _selection(self, root: APVNode, search_path: List[APVNode]) -> APVNode:
        node = root
        search_path.append(node)
        while node.children != []:
            node = node.best_child()
            search_path.append(node)
        return node

    def _expansion(self, selected_node: APVNode, search_path: List[APVNode]) -> float:
        if selected_node.is_terminal():
            return selected_node.game.get_winner()

        encoded_state = selected_node.game.encode_state()
        encoded_state = np.expand_dims(encoded_state, axis=0)
        tf_encoded_state = tf.convert_to_tensor(encoded_state, dtype=tf.float16)
        policy_tensor, value_tensor = self._predict(tf_encoded_state)

        policy = policy_tensor.numpy()
        value = value_tensor.numpy()[0]

        normalised_p = self._normalise_policy(policy, selected_node.game)
        if selected_node == self.root and self.training:
            noise = np.random.dirichlet(0.03 * np.ones(self.policy_size))
            epsilon = 0.25
            normalised_p = (1 - epsilon) * normalised_p + epsilon * noise

        selected_node.populate_children(normalised_p)
        return value

    def _backpropagation(self, result: float, search_path: List[APVNode]) -> None:
        last_node = search_path[-1]
        value = abs(result) if (last_node.game.current_player * result) < 0 else -abs(result)
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _normalise_policy(self, policy: np.ndarray, game: Game) -> np.ndarray:
        # Mask illegal moves
        masked_policy = policy * game.legal_moves_mask(self.policy_size)
        sum_masked = np.sum(masked_policy)
        # Normalise to sum to 1
        normalised_policy = masked_policy / sum_masked
        return normalised_policy

    @tf.function
    def _predict(self, inputs) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.model(inputs)
