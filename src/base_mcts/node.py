from typing import Optional, List
from math import sqrt, log
from src.game import Game


class Node:
    def __init__(
        self,
        game: Game,
        parent: Optional["Node"] = None,
        move: int = None,
    ):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = None
        self.wins = 0
        self.simulations = 0

    def is_fully_explored(self) -> bool:
        return self.children is not None and all(
            child.simulations > 0 for child in self.children
        )

    def is_terminal(self) -> bool:
        return self.game.is_game_over()

    def _uct(self) -> float:
        if self.simulations == 0 or self.parent is None:
            return float("inf")
        exploitation = self.wins / self.simulations
        exploration = sqrt(2 * log(self.parent.simulations) / self.simulations)
        return exploitation + exploration

    def best_child(self) -> "Node":
        if self.children is None:
            return self
        return max(self.children, key=lambda node: node._uct())

    def get_children(self) -> List["Node"]:
        if self.children is None:
            self.children = []
            legal_moves = self.game.get_legal_moves()
            for move in legal_moves:
                child = self.game.copy()
                child.make_move(move)
                self.children.append(Node(child, self, move))
        return self.children

    def copy(self) -> "Node":
        node = Node(self.game.copy(), self.parent, self.move)
        node.children = self.children
        node.wins = self.wins
        node.simulations = self.simulations
        return node
