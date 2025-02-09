from src.games.game import Game
from typing import List, Tuple


class Hex(Game):
    def __init__(self) -> None:
        super().__init__(11, 11)
        self.turn = 0
        self._legal_moves_cache = None

    def create_game(self) -> "Hex":
        return Hex()

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        if self.turn == 3:
            if self._legal_moves_cache and (-1, -1) in self._legal_moves_cache:
                self._legal_moves_cache.remove((-1, -1))

        if self._legal_moves_cache:
            if self.turn == 2:
                self._legal_moves_cache.append((-1, -1))
            return self._legal_moves_cache

        self._legal_moves_cache = []
        for i in range(11):
            for j in range(11):
                if self.state[i][j] == 0:
                    self._legal_moves_cache.append((i, j))
        if self.turn == 2:
            self._legal_moves_cache.append((-1, -1))
        return self._legal_moves_cache

    def make_move(self, row: int, col: int) -> None:
        if self.turn == 2 and row == -1 and col ==-1:
            for i in range(11):
                for j in range(11):
                    if self.state[i][j] != 0:
                        self.state[i][j] = self.current_player
        else:
            self.state[row][col] = self.current_player

        if self._legal_moves_cache:
            self._legal_moves_cache.remove((row, col))
        self.current_player = -self.current_player
        self.turn += 1

    def is_game_over(self) -> bool:
        # Game requires at least 21 turns to finish
        if self.turn < 21:
            return False
        return self.get_winner() != 0

    def get_winner(self) -> int:
        n = 11  # Board size
        visited = set()

        # Determine the edges to check based on the player
        start_positions = []
        self.current_player *= -1
        if self.current_player == 1:  # Red connects top to bottom
            start_positions = [(0, col) for col in range(n) if self.state[0][col] == self.current_player]
            goal_check = lambda x, y: x == n - 1  # Reached bottom row
        else:  # Blue connects left to right
            start_positions = [(row, 0) for row in range(n) if self.state[row][0] == self.current_player]
            goal_check = lambda x, y: y == n - 1  # Reached right column

        # Directions for Hex adjacency
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]

        # Depth First Search (DFS)
        def dfs(x, y):
            if goal_check(x, y):  # Check if we've reached the goal edge
                return True

            visited.add((x, y))
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in visited:
                    if self.state[nx][ny] == self.current_player and dfs(nx, ny):
                        return True
            return False

        # Start DFS from all valid starting positions
        for x, y in start_positions:
            if dfs(x, y):
                self.current_player *= -1
                return self.state[x][y]
        self.current_player *= -1
        return 0

    def is_legal_move(self, row: int, col: int) -> bool:
        return self.state[row][col] == 0

    def copy(self) -> "Hex":
        new_game = self.create_game()
        new_game.set_state([row.copy() for row in self.state])
        new_game.set_player(self.current_player)
        new_game.turn = self.turn
        new_game._legal_moves_cache = self._legal_moves_cache
        return new_game

    def reset(self) -> None:
        self.state = [[0 for _ in range(11)] for _ in range(11)]
        self.set_player(1)
        self._legal_moves_cache = None
