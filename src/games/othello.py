from src.games.game import Game
from typing import List, Tuple


class Othello(Game):
    def __init__(self) -> None:
        super().__init__(8, 8)
        self.state[3][3] = -1
        self.state[3][4] = 1
        self.state[4][3] = 1
        self.state[4][4] = -1
        self._legal_moves_cache = None

    def create_game(self) -> "Othello":
        return Othello()

    def make_move(self, row: int, col: int) -> None:
        self._legal_moves_cache = None
        if row == -1 and col == -1:
            self.current_player = -self.current_player
            return
        self.state[row][col] = self.current_player
        self.flip_pieces(row, col)
        self.current_player = -self.current_player

    def flip_pieces(self, row: int, col: int) -> None:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                self.flip_direction(row, col, i, j)

    def flip_direction(self, row: int, col: int, i: int, j: int) -> None:
        new_row = row + i
        new_col = col + j
        while (
            0 <= new_row < 8
            and 0 <= new_col < 8
            and self.state[new_row][new_col] == -self.current_player
        ):
            new_row += i
            new_col += j
        if (
            0 <= new_row < 8
            and 0 <= new_col < 8
            and self.state[new_row][new_col] == self.current_player
        ):
            while row != new_row - i or col != new_col - j:
                new_row -= i
                new_col -= j
                self.state[new_row][new_col] = self.current_player

    def get_winner(self) -> int:
        if not self.is_game_over():
            return 0
        count = 0
        for row in self.state:
            pieces = row.count(1) - row.count(-1)
            count += pieces
            if count > 32 or count < -32:
                return 1 if count > 0 else -1
        return 1 if count > 0 else -1 if count < 0 else 0

    def is_game_over(self) -> bool:
        if not self.get_possible_moves():
            self.current_player = -self.current_player
            if not self.get_possible_moves():
                return True
            self.current_player = -self.current_player
        return False

    def get_possible_moves(self) -> List[Tuple[int, int]]:
        if self._legal_moves_cache is not None:
            return self._legal_moves_cache
        self._legal_moves_cache = []
        for i in range(8):
            for j in range(8):
                if self.is_legal_move(i, j):
                    self._legal_moves_cache.append((i, j))
        return self._legal_moves_cache

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        possible_moves = self.get_possible_moves()
        if len(possible_moves) == 0:
            return [(-1, -1)]
        return possible_moves

    def is_legal_move(self, row: int, col: int) -> bool:
        if self.state[row][col] != 0:
            return False
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if self.is_legal_direction(row, col, i, j):
                    return True
        return False

    def is_legal_direction(self, row: int, col: int, i: int, j: int) -> bool:
        row += i
        col += j
        if (
            not (0 <= row < 8 and 0 <= col < 8)
            or self.state[row][col] != -self.current_player
        ):
            return False
        while (
            0 <= row < 8
            and 0 <= col < 8
            and self.state[row][col] == -self.current_player
        ):
            row += i
            col += j
        return (
            0 <= row < 8
            and 0 <= col < 8
            and self.state[row][col] == self.current_player
        )

    def reset(self) -> None:
        self.state = [[0 for _ in range(8)] for _ in range(8)]
        self.state[3][3] = -1
        self.state[3][4] = 1
        self.state[4][3] = 1
        self.state[4][4] = -1
        self.set_player(1)
        self._legal_moves_cache = None
