from src.games.game import Game
from typing import List, Tuple


class TicTacToe(Game):
    def __init__(self) -> None:
        super().__init__(3, 3)

    def create_game(self) -> "TicTacToe":
        return TicTacToe()

    def make_move(self, row: int, col: int) -> None:
        if self.state[row][col] == 0:
            self.state[row][col] = self.current_player
            self.current_player = -self.current_player

    def get_winner(self) -> int:
        for i in range(3):
            if self.state[i][0] == self.state[i][1] == self.state[i][2] != 0:
                return self.state[i][0]
            if self.state[0][i] == self.state[1][i] == self.state[2][i] != 0:
                return self.state[0][i]

        if self.state[0][0] == self.state[1][1] == self.state[2][2] != 0:
            return self.state[0][0]
        if self.state[0][2] == self.state[1][1] == self.state[2][0] != 0:
            return self.state[0][2]

        return 0

    def is_game_over(self) -> bool:
        if self.get_winner() != 0:
            return True
        return all(all(cell != 0 for cell in row) for row in self.state)

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(3) for j in range(3) if self.is_legal_move(i, j)]

    def is_legal_move(self, row: int, col: int) -> bool:
        return self.state[row][col] == 0

    def reset(self) -> None:
        self.state = [[0 for _ in range(3)] for _ in range(3)]
        self.set_player(1)
