from typing import Optional, Tuple
from math import sqrt
from src.games.game import Game


EPS = 1e-8
c_puct = 1.0


class APVNode:
    def __init__(
        self,
        game: Game,
        parent: Optional["APVNode"] = None,
        action: int = None,
        prior: int = 0,
    ) -> None:
        self.game = game
        self.parent = parent
        self.action = action
        self.P = prior

        self.children = []
        self.Qs = 0.0
        self.Ns = 0

        self.terminal = None

    def _puct_score(self) -> float:
        # Unexplored nodes have maximum priority
        if not self.parent:
            return float("inf")

        if self.Ns > 0:
            u_score = (self.Qs / self.Ns) + c_puct * self.P * sqrt(self.parent.Ns) / (1 + self.Ns)
        else:
            u_score = c_puct * self.P * sqrt(self.parent.Ns + EPS)

        return u_score

    def best_child(self) -> "APVNode":
        if not self.children:
            return self
        return max(self.children, key=lambda node: node._puct_score())

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
                prior = normalised_p[0, move]
                self.children.append(APVNode(child, self, move, prior))
