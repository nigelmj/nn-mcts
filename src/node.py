import random
from math import sqrt
from typing import Optional

from src.game import Game

EPS = 1e-8
c_puct = 1.0


class Node:
    def __init__(
        self,
        game: Game,
        parent: Optional["Node"] = None,
        action: int = -1,
        prior: int = 0,
    ) -> None:
        self.game = game
        self.parent = parent
        self.action = action
        self.P = prior

        self.children = {}
        self.Qs = 0.0
        self.Ns = 0

        self.terminal = None

    def _puct_score(self) -> float:
        # Unexplored nodes have maximum priority
        if not self.parent:
            return float("inf")

        if self.Ns > 0:
            u_score = (self.Qs / self.Ns) + c_puct * self.P * sqrt(self.parent.Ns) / (
                1 + self.Ns
            )
        else:
            u_score = c_puct * self.P * sqrt(self.parent.Ns + EPS)

        return u_score

    def best_child(self) -> "Node":
        if not self.children:
            return self
        max_score = max(node._puct_score() for node in self.children.values())
        best_nodes = [
            node for node in self.children.values() if node._puct_score() == max_score
        ]
        return random.choice(best_nodes)

    def is_terminal(self) -> bool:
        if self.terminal is not None:
            return self.terminal
        self.terminal = self.game.is_game_over()
        return self.terminal

    def populate_children(self, normalised_p) -> None:
        if not self.children:
            for move in self.game.get_legal_moves():
                child = self.game.copy()
                child.make_move(move)
                prior = normalised_p[move]
                self.children[move] = Node(child, self, move, prior)
