from src.game import Game
import numpy as np


class TicTacToe(Game):
    def __init__(self) -> None:
        super().__init__(3, 3, 9)

    def create_game(self) -> "TicTacToe":
        return TicTacToe()

    def make_move(self, action: int) -> None:
        row, col = divmod(action, self.size2)
        self.state[row, col] = self.current_player
        self.current_player = -self.current_player

    def get_winner(self) -> int:
        row_win = np.all(self.state == self.state[:, [0]], axis=1) & (self.state[:, 0] != 0)
        if np.any(row_win):
            return self.state[row_win, 0][0]

        col_win = np.all(self.state == self.state[[0], :], axis=0) & (self.state[0, :] != 0)
        if np.any(col_win):
            return self.state[0, col_win][0]

        diag1 = np.diag(self.state)
        if np.all(diag1 == diag1[0]) and diag1[0] != 0:
            return diag1[0]

        diag2 = np.diag(np.fliplr(self.state))
        if np.all(diag2 == diag2[0]) and diag2[0] != 0:
            return diag2[0]

        return 0

    def is_game_over(self) -> bool:
        if self.get_winner() != 0:
            return True
        return np.all(self.state)

    def get_legal_moves(self) -> np.ndarray:
        return np.flatnonzero(self.state == 0)

    def is_legal_move(self, action: int) -> bool:
        row, col = divmod(action, self.size2)
        return self.state[row,col] == 0

    def reset(self) -> None:
        self.state = np.zeros((self.size1, self.size2))
        self.set_player(1)
