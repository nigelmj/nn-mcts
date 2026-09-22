from typing import List, Sequence, Tuple

import numpy as np
import torch

from src.node import Node


def apply_temperature(policy: np.ndarray, temperature: float) -> np.ndarray:
    """Sharpen or flatten a visit distribution for MOVE SELECTION only.

    Temperature 1 returns the distribution unchanged, temperature 0 returns a one-hot on
    the most visited action (ties broken at random), and values in between interpolate.

    This is deliberately separate from the distribution stored as a training target. The
    target must always stay at temperature 1: it is the search's considered opinion over
    every move, and collapsing it to a one-hot throws away most of what the search
    learned. Selection is a different question -- late in a game you want the best move,
    not a sample -- so it gets its own knob.
    """
    policy = np.asarray(policy, dtype=np.float64)

    def greedy() -> np.ndarray:
        best = np.argwhere(policy == policy.max()).flatten()
        out = np.zeros_like(policy)
        out[np.random.choice(best)] = 1.0
        return out

    if temperature <= 0:
        return greedy()

    total = policy.sum()
    if total <= 0:
        raise ValueError("cannot apply a temperature to an all-zero policy")
    if temperature == 1.0:
        return policy / total

    scaled = np.power(policy, 1.0 / temperature)
    scaled_total = scaled.sum()
    if scaled_total <= 0 or not np.isfinite(scaled_total):
        # Very low temperatures underflow small probabilities to zero. Greedy is the
        # limit this is heading towards anyway, so take it rather than emit NaNs.
        return greedy()
    return scaled / scaled_total


def temperature_for_move(
    schedule: Sequence[Tuple[int, float]], move_count: int
) -> float:
    """Look up the temperature for a ply in a [(from_move, temperature), ...] schedule.

    The schedule is a step function: each entry sets the temperature from that ply until
    the next entry. [(0, 1.0), (15, 0.5), (30, 0.0)] samples freely for the first fifteen
    plies, then more sharply, then plays greedily from ply thirty.
    """
    if not schedule:
        return 0.0
    ordered = sorted(schedule, key=lambda entry: entry[0])
    temperature = ordered[0][1]
    for from_move, value in ordered:
        if move_count >= from_move:
            temperature = value
        else:
            break
    return float(temperature)


class MCTS:
    def __init__(
        self,
        root: Node,
        num_simulations: int,
        training: bool,
        parallelised: bool,
        **kwargs,
    ) -> None:
        self.root = root
        self.num_simulations = num_simulations
        self.training = training
        self.parallelised = parallelised
        if parallelised:
            self.request_queue = kwargs["request_queue"]
            self.response_queue = kwargs["response_queue"]
            self.wid = kwargs["wid"]
        else:
            self.model = kwargs["model"]

    def compute_improved_policy(self) -> np.ndarray:
        """The search's visit distribution over actions, normalised to sum to 1.

        Always temperature 1. Callers that want a single move apply their own temperature
        via apply_temperature; callers storing a training target use this as-is.
        """
        for _ in range(self.num_simulations):
            search_path = []
            node = self._selection(self.root, search_path)
            value = self._expansion(node)
            self._backpropagation(value, search_path)

        total_visits = sum(child.Ns for child in self.root.children.values())
        policy = np.zeros(self.root.game.policy_size)
        if total_visits <= 0:
            raise RuntimeError(
                "search expanded no children; the root is terminal or "
                "num_simulations is too small"
            )

        for child in self.root.children.values():
            policy[child.action] = child.Ns / total_visits

        return policy / np.sum(policy)

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
