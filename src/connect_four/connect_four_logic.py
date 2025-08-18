from src.game import Game
import numpy as np
from scipy.signal import convolve2d


class ConnectFour(Game):
    def __init__(self) -> None:
        super().__init__(6, 7, 42)

    def create_game(self) -> "ConnectFour":
        return ConnectFour()

    def make_move(self, action: int) -> None:
        row, col = divmod(action, self.size2)
        self.state[row,col] = self.current_player
        self.current_player = -self.current_player

    def get_winner(self) -> int:
        for player in [1, -1]:
            player_board = (self.state == player).astype(int)

            kernel_h = np.ones((1, 4), dtype=int)
            if np.any(convolve2d(player_board, kernel_h, mode='valid') == 4):
                return player

            kernel_v = np.ones((4, 1), dtype=int)
            if np.any(convolve2d(player_board, kernel_v, mode='valid') == 4):
                return player

            kernel_d1 = np.eye(4, dtype=int)
            if np.any(convolve2d(player_board, kernel_d1, mode='valid') == 4):
                return player

            flipped = np.flipud(player_board)
            if np.any(convolve2d(flipped, kernel_d1, mode='valid') == 4):
                return player

        return 0

    def is_game_over(self) -> bool:
        if self.get_winner() != 0:
            return True
        return np.all(self.state)

    def get_legal_moves(self) -> np.ndarray:
        rows, cols = self.state.shape
        legal_columns = [col for col in range(cols) if self.state[0, col] == 0]

        # Convert legal columns to flattened indices
        flattened_indices = [(rows - 1 - np.count_nonzero(self.state[:, col])) * cols + col for col in legal_columns]
        return np.array(flattened_indices)

    def is_legal_move(self, action: int) -> bool:
        row, col = divmod(action, self.size2)
        return self.state[row,col] == 0 and (row == 5 or self.state[row + 1,col] != 0)

    def reset(self) -> None:
        self.state = np.zeros((self.size1, self.size2))
        self.set_player(1)
