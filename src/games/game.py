from abc import ABC, abstractmethod
import random
import numpy as np
from typing import List, Tuple


class Game(ABC):
    def __init__(self, size1: int, size2: int) -> None:
        self.size1 = size1
        self.size2 = size2
        self.state = [[0 for _ in range(self.size2)] for _ in range(self.size1)]
        self.set_player(1)

    @abstractmethod
    def make_move(self, row: int, col: int) -> None:
        pass

    @abstractmethod
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def is_legal_move(self, row: int, col: int) -> bool:
        pass

    def make_random_move(self) -> Tuple[int, int]:
        return random.choice(self.get_legal_moves())

    @abstractmethod
    def get_winner(self) -> int:
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        pass

    @abstractmethod
    def create_game(self) -> "Game":
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def set_state(self, state: List[List[int]]) -> None:
        self.state = state

    def set_player(self, player: int) -> None:
        self.current_player = player

    def copy(self) -> "Game":
        new_game = self.create_game()
        new_game.set_state([row.copy() for row in self.state])
        new_game.set_player(self.current_player)
        return new_game

    def encode_state(self) -> np.ndarray:
        state = np.array(self.state)
        player_channel = (state == self.current_player).astype(int)
        opponent_channel = (state == -self.current_player).astype(int)
        return np.stack([player_channel, opponent_channel], axis=0)

    def legal_moves_mask(self, policy_size) -> np.ndarray:
        mask = np.zeros(policy_size, dtype=np.float32)
        for move in self.get_legal_moves():
            if move == (-1, -1):
                mask[-1] = 1.0
            else:
                idx = move[0] * self.size2 + move[1]
                mask[idx] = 1.0
        return mask
