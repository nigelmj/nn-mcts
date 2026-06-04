from abc import ABC, abstractmethod

import numpy as np


class Game(ABC):
    def __init__(self, size1: int, size2: int, policy_size: int) -> None:
        self.size1 = size1
        self.size2 = size2
        self.state = np.zeros((self.size1, self.size2))
        self.policy_size = policy_size
        self.set_player(1)
        self.move_number = 1

    @abstractmethod
    def make_move(self, action: int) -> None:
        pass

    @abstractmethod
    def get_legal_moves(self) -> np.ndarray:
        pass

    @abstractmethod
    def is_legal_move(self, action: int) -> bool:
        pass

    def get_random_move(self) -> int:
        legal_moves = self.get_legal_moves()
        chosen_move = legal_moves[np.random.choice(len(legal_moves))]
        return chosen_move

    def make_random_move(self) -> None:
        action = self.get_random_move()
        self.make_move(action)

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

    def set_state(self, state: np.ndarray) -> None:
        self.state = state

    def set_player(self, player: int) -> None:
        self.current_player = player

    def copy(self) -> "Game":
        new_game = self.create_game()
        new_game.set_state(np.copy(self.state))
        new_game.set_player(self.current_player)
        return new_game

    def encode_state(self) -> np.ndarray:
        current_player_channel = (self.state == self.current_player).astype(int)
        opponent_player_channel = (self.state == -self.current_player).astype(int)
        return np.stack([current_player_channel, opponent_player_channel], axis=0)

    def legal_moves_mask(self) -> np.ndarray:
        mask = np.zeros(self.policy_size, dtype=bool)
        mask[self.get_legal_moves()] = True
        return mask

    def mask_normalise_policy(self, policy: np.ndarray) -> np.ndarray:
        # Mask illegal moves
        masked_policy = policy * self.legal_moves_mask()
        sum_masked = np.sum(masked_policy)

        normalised_policy = masked_policy / sum_masked
        return normalised_policy
