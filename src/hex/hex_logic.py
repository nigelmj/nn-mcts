import numpy as np

from src.game import Game


class Hex(Game):
    def __init__(self) -> None:
        super().__init__(11, 11, 122)
        self.move_number = 1
        self.pie_rule_used = False

    def create_game(self) -> "Hex":
        return Hex()

    def get_legal_moves(self) -> np.ndarray:
        legal_moves = np.flatnonzero(self.state == 0)
        if self.move_number == 2:
            return np.append(legal_moves, self.size1 * self.size2)
        return legal_moves

    def make_move(self, action: int) -> None:
        if action == self.size1 * self.size2 and self.move_number == 2:
            self.state = -self.state
            self.pie_rule_used = True
        else:
            row, col = divmod(action, self.size2)
            self.state[row, col] = self.current_player
        self.current_player = -self.current_player
        self.move_number += 1

    def is_game_over(self) -> bool:
        # Game requires both players to play at least num_size turns each to finish
        if self.move_number < (self.size1 + self.size2 - 1):
            return False
        return self.get_winner() != 0

    def get_winner(self) -> int:
        n = self.size1
        visited = set()

        # Determine the edges to check based on the player
        start_positions = []
        prev_move_player = self.current_player * -1
        if prev_move_player == 1:  # Player 1 connects top to bottom
            start_positions = [
                (0, col) for col in range(n) if self.state[0, col] == prev_move_player
            ]
            goal_check = lambda x, y: x == n - 1
        else:  # Player 2 connects left to right
            start_positions = [
                (row, 0) for row in range(n) if self.state[row, 0] == prev_move_player
            ]
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
        return (
            action == self.size1 * self.size2 and self.move_number == 2
        ) or self.state[row, col] == 0

    def copy(self) -> "Hex":
        new_game = self.create_game()
        new_game.set_state(np.copy(self.state))
        new_game.set_player(self.current_player)
        new_game.move_number = self.move_number
        new_game.pie_rule_used = self.pie_rule_used
        return new_game

    def reset(self) -> None:
        self.state = np.zeros((self.size1, self.size2))
        self.set_player(1)
        self.move_number = 1
        self.pie_rule_used = False

    def encode_state(self) -> np.ndarray:
        if self.current_player == 1:
            return super().encode_state()

        # Flip on one diagonal so network always sees current
        # player going the top-bottom direction even if second
        # player in reality traverses the left-right route
        encoded_state = super().encode_state()
        encoded_state = np.rot90(encoded_state, k=1, axes=(1, 2))
        encoded_state = np.flip(encoded_state, axis=2)
        return encoded_state

    def mask_normalise_policy(self, policy: np.ndarray) -> np.ndarray:
        if self.current_player == 1:
            return super().mask_normalise_policy(policy)

        swap_move = policy[-1]
        policy = np.delete(policy, -1)
        policy_2d = policy.reshape(self.size1, self.size2)

        policy_2d = np.flip(policy_2d, axis=1)
        policy_2d = np.rot90(policy_2d, k=3)
        transfomed_policy = policy_2d.flatten()
        transformed_policy = np.append(transfomed_policy, swap_move)

        return super().mask_normalise_policy(transformed_policy)
