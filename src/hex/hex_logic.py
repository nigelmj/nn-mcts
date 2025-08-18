from src.game import Game
import numpy as np


class Hex(Game):
    def __init__(self) -> None:
        super().__init__(7, 7, 50)
        self.turn = 1
        self.pie_rule_used = False

    def create_game(self) -> "Hex":
        return Hex()

    def get_legal_moves(self) -> np.ndarray:
        legal_moves = np.flatnonzero(self.state == 0)
        if self.turn == 2:
            return np.append(legal_moves, self.size1 * self.size2)
        return legal_moves

    def make_move(self, action: int) -> None:
        if action == self.size1 * self.size2 and self.turn == 2:
            self.state = -self.state
            self.pie_rule_used = True
        else:
            row, col = divmod(action, self.size2)
            self.state[row,col] = self.current_player
        self.current_player = -self.current_player
        self.turn += 1

    def is_game_over(self) -> bool:
        # Game requires both players to play at least num_size turns each to finish
        if self.turn < (self.size1 + self.size2 - 1):
            return False
        return self.get_winner() != 0

    def get_winner(self) -> int:
        n = self.size1
        visited = set()

        # Determine the edges to check based on the player
        start_positions = []
        prev_move_player = self.current_player * -1
        if prev_move_player == 1:  # Player 1 connects top to bottom
            start_positions = [(0, col) for col in range(n) if self.state[0, col] == prev_move_player]
            goal_check = lambda x, y: x == n - 1
        else:  # Player 2 connects left to right
            start_positions = [(row, 0) for row in range(n) if self.state[row, 0] == prev_move_player]
            goal_check = lambda x, y: y == n - 1

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]

        def dfs(x, y):
            if goal_check(x, y):
                return True

            visited.add((x, y))
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in visited:
                    if self.state[nx, ny] == prev_move_player and dfs(nx, ny):
                        return True
            return False

        for x, y in start_positions:
            if dfs(x, y):
                return prev_move_player

        return 0

    def is_legal_move(self, action: int) -> bool:
        row, col = divmod(action, self.size2)
        return (action == self.size1 * self.size2 and self.turn == 2) or self.state[row,col] == 0

    def copy(self) -> "Hex":
        new_game = self.create_game()
        new_game.set_state(np.copy(self.state))
        new_game.set_player(self.current_player)
        new_game.turn = self.turn
        new_game.pie_rule_used = self.pie_rule_used
        return new_game

    def reset(self) -> None:
        self.state = np.zeros((self.size1, self.size2))
        self.set_player(1)
        self.turn = 1
        self.pie_rule_used = False

    def encode_state(self) -> np.ndarray:
        first_player_plane = (self.state == 1).astype(int)
        second_player_plane = (self.state == -1).astype(int)
        turn_indicator_plane = np.ones_like(first_player_plane) \
                            if (self.current_player == 1) \
                            else np.zeros_like(first_player_plane)
        swap_move_plane = np.ones_like(first_player_plane) \
                                    if (self.pie_rule_used) \
                                    else np.zeros_like(first_player_plane)
        return np.stack([first_player_plane, second_player_plane, turn_indicator_plane, swap_move_plane], axis=0)
