from typing import List

import numpy as np
import torch

from src.node import Node


class MCTS:
    def __init__(
        self,
        root: Node,
        num_simulations: int,
        training: bool,
        sampling: bool,
        parallelised: bool,
        **kwargs,
    ) -> None:
        self.root = root
        self.num_simulations = num_simulations
        self.training = training
        self.sampling = sampling
        self.parallelised = parallelised
        if parallelised:
            self.request_queue = kwargs["request_queue"]
            self.response_queue = kwargs["response_queue"]
            self.wid = kwargs["wid"]
        else:
            self.model = kwargs["model"]

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
            return -(
                selected_node.game.get_winner() * selected_node.game.current_player
            )

        encoded_state = selected_node.game.encode_state()
        if self.parallelised:
            self.request_queue.put((self.wid, encoded_state))
            policy, value = self.response_queue.get()
        else:
            tensor_state = (
                torch.from_numpy(encoded_state.copy())
                .float()
                .unsqueeze(0)
                .to(next(self.model.parameters()).device)
            )
            policy, value = self.model.predict(tensor_state)
            policy = policy[0]

        normalised_p = selected_node.game.mask_normalise_policy(policy)
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
