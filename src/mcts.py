from src.game import Game
from src.node import Node
from src.neural_network import AlphaZeroNetwork
import torch
import numpy as np
from typing import List


class MCTS:
    def __init__(self,
        root: Node,
        model: AlphaZeroNetwork,
        num_simulations: int,
        training: bool,
        sampling: bool,
    ) -> None:
        self.root = root
        self.model = model
        self.num_simulations = num_simulations
        self.training = training
        self.sampling = sampling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_improved_policy(self) -> np.ndarray:
        for _ in range(self.num_simulations):
            search_path = []
            node = self._selection(self.root, search_path)
            value = self._expansion(node)
            self._backpropagation(value, search_path)

        total_visits = sum(child.Ns for child in self.root.children.values())
        policy = np.zeros(self.root.game.policy_size)

        for child in self.root.children.values():
            policy[child.action] = child.Ns / total_visits

        # print(policy)
        if not self.sampling:
            best_actions = np.argwhere(policy == np.max(policy)).flatten()
            best_action = np.random.choice(best_actions)
            policy = np.zeros(self.root.game.policy_size)
            policy[best_action] = 1
            return policy

        policy /= np.sum(policy)
        return policy

    def _selection(self, root, search_path: List) -> Node:
        node = root
        search_path.append(node)
        while node.children != {}:
            node = node.best_child()
            search_path.append(node)
        return node

    def _expansion(self, selected_node: Node) -> float:
        if selected_node.is_terminal():
            # Get winner is from a global perspective, change to the perspective of the current player
            # Negate it since the current player is one that has to make a move, but the game has been won
            # by the player that just made the move.
            return -(selected_node.game.get_winner() * selected_node.game.current_player)

        encoded_state = selected_node.game.encode_state()
        encoded_state = np.expand_dims(encoded_state, axis=0)

        # Convert to PyTorch tensor
        state_tensor = torch.tensor(encoded_state, dtype=torch.float32).to(self.device)
        policy, value = self.model.predict(state_tensor)

        normalised_p = self._mask_normalise_policy(policy, selected_node.game)
        if selected_node == self.root and self.training:
            noise = np.random.dirichlet(0.03 * np.ones(selected_node.game.policy_size))
            epsilon = 0.25
            normalised_p = (1 - epsilon) * normalised_p + epsilon * noise

        selected_node.populate_children(normalised_p)

        # Negation of value since predicted value is for next player,
        # but node is for current player
        return -value

    def _backpropagation(self, value: float, search_path: List) -> None:
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
