from src.games.game import Game
from typing import List, Tuple


class Connect4(Game):
    def __init__(self) -> None:
        super().__init__(6, 7)

    def create_game(self) -> "Connect4":
        return Connect4()

    def make_move(self, row: int, col: int) -> None:
        self.state[row][col] = self.current_player
        self.current_player = -self.current_player

    def get_winner(self) -> int:
        for row in range(6):
            for col in range(7):
                if self.state[row][col] != 0:
                    if col + 3 < 7 and all(
                        self.state[row][col + i] == self.state[row][col]
                        for i in range(4)
                    ):
                        return self.state[row][col]

                    if row + 3 < 6 and all(
                        self.state[row + i][col] == self.state[row][col]
                        for i in range(4)
                    ):
                        return self.state[row][col]

                    if (
                        row + 3 < 6
                        and col + 3 < 7
                        and all(
                            self.state[row + i][col + i] == self.state[row][col]
                            for i in range(4)
                        )
                    ):
                        return self.state[row][col]

                    if (
                        row - 3 >= 0
                        and col + 3 < 7
                        and all(
                            self.state[row - i][col + i] == self.state[row][col]
                            for i in range(4)
                        )
                    ):
                        return self.state[row][col]
        return 0

    def is_game_over(self) -> bool:
        if self.get_winner() != 0:
            return True
        return all(all(cell != 0 for cell in row) for row in self.state)

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        moves = []
        for col in range(7):
            for row in range(5, -1, -1):
                if self.state[row][col] == 0:
                    moves.append((row, col))
                    break
        return moves

    def is_legal_move(self, row: int, col: int) -> bool:
        return self.state[row][col] == 0 and (row == 5 or self.state[row + 1][col] != 0)

    def reset(self) -> None:
        self.state = [[0 for _ in range(7)] for _ in range(6)]
        self.set_player(1)
