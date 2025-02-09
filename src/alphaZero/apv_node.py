from typing import Optional, Tuple
from math import sqrt
from src.games.game import Game
import random


c = 1.0


class APVNode:
    def __init__(
        self,
        game: Game,
        parent: Optional["APVNode"] = None,
        action: Optional[Tuple[int, int]] = None,
        prior_probability: int = 0,
    ) -> None:
        self.game = game
        self.parent = parent
        self.action = action
        self.prior_probability = prior_probability

        self.children = []
        self.value_sum = 0.0
        self.visit_count = 0

        self.terminal = None

    def _puct_score(self) -> float:
        # Unexplored nodes have maximum priority
        if self.visit_count == 0:
            return float("inf")

        top_node = self
        if self.parent:
            top_node = self.parent

        q_score = self.value_sum / self.visit_count
        u_score = (
            c
            * self.prior_probability
            * sqrt(top_node.visit_count)
            / (1 + self.visit_count)
        )
        return q_score + u_score

    def best_child(self) -> "APVNode":
        if not self.children:
            return self

        child_scores = [(child, child._puct_score()) for child in self.children]
        max_score = max(score for _, score in child_scores)
        best_candidates = [child for child, score in child_scores if score == max_score]
        return random.choice(best_candidates)

    def is_terminal(self) -> bool:
        if self.terminal is not None:
            return self.terminal
        self.terminal = self.game.is_game_over()
        return self.terminal

    def populate_children(self, normalised_p) -> None:
        if not self.children:
            for move in self.game.get_legal_moves():
                i, j = move
                child = self.game.copy()
                child.make_move(i, j)

                if move == (-1, -1):
                    idx = -1
                else:
                    idx = i * self.game.size2 + j
                prior_probability = normalised_p[0, idx]

                self.children.append(APVNode(child, self, move, prior_probability))
