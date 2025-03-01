from src.games.game import Game
from src.alphaZero.apv_node import APVNode
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
from typing import List, Tuple


class APVMCTS:
    def __init__(self,
        root: APVNode,
        model: Model,
        num_simulations: int,
        training: bool,
        sampling: bool,
    ) -> None:
        self.root = root
        self.model = model
        self.num_simulations = num_simulations
        self.training = training
        self.sampling = sampling

    def compute_improved_policy(self) -> np.ndarray:
        for _ in range(self.num_simulations):
            search_path = []
            node = self._selection(self.root, search_path)
            result = self._expansion(node)
            self._backpropagation(result, search_path)

        total_visits = sum(child.Ns for child in self.root.children)
        policy = np.zeros(self.root.game.policy_size)

        for child in self.root.children:
            policy[child.action] = child.Ns / total_visits

        if not self.sampling:
            best_actions = np.argwhere(policy == np.max(policy)).flatten()
            best_action = np.random.choice(best_actions)
            policy = np.zeros(self.root.game.policy_size)
            policy[best_action] = 1
            return policy
            
        policy /= np.sum(policy)
        return policy

    def _selection(self, root: APVNode, search_path: List[APVNode]) -> APVNode:
        node = root
        search_path.append(node)
        while node.children != []:
            node = node.best_child()
            search_path.append(node)
        return node

    def _expansion(self, selected_node: APVNode) -> float:
        if selected_node.is_terminal():
            return selected_node.game.get_winner()

        encoded_state = selected_node.game.encode_state()
        encoded_state = np.expand_dims(encoded_state, axis=0)
        tf_encoded_state = tf.convert_to_tensor(encoded_state, dtype=tf.float64)
        policy_tensor, value_tensor = self._predict(tf_encoded_state)

        policy = policy_tensor.numpy()
        value = value_tensor.numpy()[0]

        normalised_p = self._mask_normalise_policy(policy, selected_node.game)
        if selected_node == self.root and self.training:
            noise = np.random.dirichlet(0.03 * np.ones(selected_node.game.policy_size))
            epsilon = 0.25
            normalised_p = (1 - epsilon) * normalised_p + epsilon * noise

        selected_node.populate_children(normalised_p)

        # Change value from player to global perspective
        return value * selected_node.game.current_player

    def _backpropagation(self, result: float, search_path: List[APVNode]) -> None:
        last_node = search_path[-1]

        # Switch back to player perspective for nodes
        value = abs(result) if (last_node.game.current_player * result) < 0 else -abs(result)
        for node in reversed(search_path):
            node.Ns += 1
            node.Qs += value
            value = -value

    def _mask_normalise_policy(self, policy: np.ndarray, game: Game) -> np.ndarray:
        # Mask illegal moves
        masked_policy = policy * game.legal_moves_mask()
        sum_masked = np.sum(masked_policy)

        normalised_policy = masked_policy / sum_masked
        return normalised_policy

    @tf.function
    def _predict(self, inputs) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.model(inputs)
